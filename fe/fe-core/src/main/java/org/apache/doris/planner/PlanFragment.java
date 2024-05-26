// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied. See the License for the
// specific language governing permissions and limitations
// under the License.
// This file is copied from
// https://github.com/apache/impala/blob/branch-2.9.0/fe/src/main/java/org/apache/impala/PlanFragment.java
// and modified by Doris

package org.apache.doris.planner;

import org.apache.doris.analysis.Expr;
import org.apache.doris.analysis.QueryStmt;
import org.apache.doris.analysis.SlotDescriptor;
import org.apache.doris.analysis.SlotRef;
import org.apache.doris.analysis.StatementBase;
import org.apache.doris.analysis.TupleDescriptor;
import org.apache.doris.common.TreeNode;
import org.apache.doris.qe.ConnectContext;
import org.apache.doris.thrift.TExplainLevel;
import org.apache.doris.thrift.TPartitionType;
import org.apache.doris.thrift.TPlanFragment;
import org.apache.doris.thrift.TResultSinkType;

import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;
import org.apache.commons.collections.CollectionUtils;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * PlanFragments通过其ExchangeNodes形成树结构。这样连接的片段树形成一个计划。
 * 计划的输出由根片段生成，是查询结果或另一个计划所需的中间结果（如哈希表）。
 *
 * 计划根据其输出的使用者进行分组：所有为特定使用者计划物化中间结果的计划都被分组到一个单独的组中。
 *
 * PlanFragment封装了用于生成计划片段输出的具体执行节点树，以及输出表达式、目标节点等。
 * 如果没有输出表达式，则标记由计划根节点生成的整个行被物化。
 *
 * 一个计划片段可以有一个或多个实例，每个实例由单个节点执行，输出发送到目标片段的特定实例（或在根片段的情况下以某种形式物化）。
 *
 * 一个哈希分区的计划片段是一个或多个哈希分区数据流被该片段中的计划节点接收的结果。
 * 将来，片段的数据分区也可以基于从物理哈希分区表读取的扫描节点进行哈希分区。
 *
 * 调用顺序是：
 * - 构造函数
 * - 使用getter等方法进行组装
 * - finalize()
 * - toThrift()
 *
 * TODO: PlanNodes的树在片段边界上连接，这使得无法在片段内搜索内容（使用TreeNode函数）；需要修复这一点
 */
public class PlanFragment extends TreeNode<PlanFragment> {
    private static final Logger LOG = LogManager.getLogger(PlanFragment.class);

    // 此计划片段的ID
    private PlanFragmentId fragmentId;
    // nereids规划器和原始规划器以不同的顺序生成片段。
    // 这使得nereids片段ID与原始规划器不同，因此与配置文件中的ID不同。
    // 在原始规划器中，fragmentSequenceNum是fragmentId，而在nereids规划器中，
    // fragmentSequenceNum是配置文件中显示的ID
    private int fragmentSequenceNum;
    // private PlanId planId_;
    // private CohortId cohortId_;

    // 此片段执行的计划树的根节点
    private PlanNode planRoot;

    // 此片段将其输出发送到的交换节点
    private ExchangeNode destNode;

    // 如果为null，则输出计划根节点生成的整个行
    private ArrayList<Expr> outputExprs;

    // 在finalize()中创建或在setSink()中设置
    protected DataSink sink;

    // 此片段中特定分区的数据源（或发送方）；一个未分区的片段仅在单个节点上执行
    private DataPartition dataPartition;

    // 传输到BE时实际输入分区的规范。
    // 默认情况下，规划器中的数据分区值和传输到BE的分区值相同。因此此属性为空。
    // 但有时计划值和序列化值不一致，需要设置此值。
    // 目前，这种情况仅发生在包含扫描节点的片段中。
    // 由于扫描节点的数据分区表达式实际上是根据表的模式构建的，因此表达式未进行分析。
    // 这将导致此表达式无法正确序列化和传输到BE。
    // 在这种情况下，需要将此属性设置为DataPartition RANDOM以避免问题。
    private DataPartition dataPartitionForThrift;

    // 此片段输出的分区规范（即，它如何发送到目标）；如果输出是未分区的，则正在广播
    protected DataPartition outputPartition;

    // 查询统计信息是否与每个批次一起发送。为了在查询包含限制时正确获取查询统计信息，有必要与每个批次一起发送查询统计信息，或仅在关闭时发送。
    private boolean transferQueryStatisticsWithEveryBatch;

    // TODO: SubstitutionMap outputSmap;
    // 替换映射，用于将表达式重新映射到此片段的输出，在目标片段应用

    // 片段执行时的并行数量规范，默认值为1
    private int parallelExecNum = 1;

    // 生成的运行时过滤器ID
    private Set<RuntimeFilterId> builderRuntimeFilterIds;
    // 预期使用的运行时过滤器ID
    private Set<RuntimeFilterId> targetRuntimeFilterIds;

    private int bucketNum;

    // 是否有共置计划节点
    protected boolean hasColocatePlanNode = false;

    private TResultSinkType resultSinkType = TResultSinkType.MYSQL_PROTOCAL;

    /**
     * 带特定分区的片段的构造函数；默认情况下，输出为广播。
     */
    public PlanFragment(PlanFragmentId id, PlanNode root, DataPartition partition) {
        this.fragmentId = id;
        this.planRoot = root;
        this.dataPartition = partition;
        this.outputPartition = DataPartition.UNPARTITIONED;
        this.transferQueryStatisticsWithEveryBatch = false;
        this.builderRuntimeFilterIds = new HashSet<>();
        this.targetRuntimeFilterIds = new HashSet<>();
        setParallelExecNumIfExists();
        setFragmentInPlanTree(planRoot);
    }

    public PlanFragment(PlanFragmentId id, PlanNode root, DataPartition partition, DataPartition partitionForThrift) {
        this(id, root, partition);
        this.dataPartitionForThrift = partitionForThrift;
    }

    public PlanFragment(PlanFragmentId id, PlanNode root, DataPartition partition,
                        Set<RuntimeFilterId> builderRuntimeFilterIds, Set<RuntimeFilterId> targetRuntimeFilterIds) {
        this(id, root, partition);
        this.builderRuntimeFilterIds = new HashSet<>(builderRuntimeFilterIds);
        this.targetRuntimeFilterIds = new HashSet<>(targetRuntimeFilterIds);
    }

    /**
     * 将“this”分配为以node为根的计划树中所有PlanNode的片段。
     * 不遍历ExchangeNode的子节点，因为这些子节点必须属于不同的片段。
     */
    public void setFragmentInPlanTree(PlanNode node) {
        if (node == null) {
            return;
        }
        node.setFragment(this);
        if (node instanceof ExchangeNode) {
            return;
        }
        for (PlanNode child : node.getChildren()) {
            setFragmentInPlanTree(child);
        }
    }

    /**
     * 根据SessionVariable中的PARALLEL_FRAGMENT_EXEC_INSTANCE_NUM分配ParallelExecNum用于同步请求
     * 对于异步请求，默认值分配ParallelExecNum
     */
    public void setParallelExecNumIfExists() {
        if (ConnectContext.get() != null) {
            parallelExecNum = ConnectContext.get().getSessionVariable().getParallelExecInstanceNum();
        }
    }

    // 手动设置并行执行数量
    // 目前用于代理加载
    public void setParallelExecNum(int parallelExecNum) {
        this.parallelExecNum = parallelExecNum;
    }

    public void setOutputExprs(List<Expr> outputExprs) {
        this.outputExprs = Expr.cloneList(outputExprs, null);
    }

    public void resetOutputExprs(TupleDescriptor tupleDescriptor) {
        this.outputExprs = Lists.newArrayList();
        for (SlotDescriptor slotDescriptor : tupleDescriptor.getSlots()) {
            SlotRef slotRef = new SlotRef(slotDescriptor);
            outputExprs.add(slotRef);
        }
    }

    public ArrayList<Expr> getOutputExprs() {
        return outputExprs;
    }

    public void setBuilderRuntimeFilterIds(RuntimeFilterId rid) {
        this.builderRuntimeFilterIds.add(rid);
    }

    public void setTargetRuntimeFilterIds(RuntimeFilterId rid) {
        this.targetRuntimeFilterIds.add(rid);
    }

    public void setHasColocatePlanNode(boolean hasColocatePlanNode) {
        this.hasColocatePlanNode = hasColocatePlanNode;
    }

    public void setResultSinkType(TResultSinkType resultSinkType) {
        this.resultSinkType = resultSinkType;
    }

    public boolean hasColocatePlanNode() {
        return hasColocatePlanNode;
    }

    public void setDataPartition(DataPartition dataPartition) {
        this.dataPartition = dataPartition;
    }

    /**
     * 完成计划树并创建流接收器（如果需要）。
     */
    public void finalize(StatementBase stmtBase) {
        if (sink != null) {
            return;
        }
        if (destNode != null) {
            Preconditions.checkState(sink == null);
            // 我们正在向交换节点流式传输
            DataStreamSink streamSink = new DataStreamSink(destNode.getId());
            streamSink.setOutputPartition(outputPartition);
            streamSink.setFragment(this);
            sink = streamSink;
        } else {
            if (planRoot == null) {
                // 仅输出表达式，没有FROM子句
                // "select 1 + 2"
                return;
            }
            Preconditions.checkState(sink == null);
            QueryStmt queryStmt = stmtBase instanceof QueryStmt ? (QueryStmt) stmtBase : null;
            if (queryStmt != null && queryStmt.hasOutFileClause()) {
                sink = new ResultFileSink(planRoot.getId(), queryStmt.getOutFileClause(), queryStmt.getColLabels());
            } else {
                // 添加ResultSink
                // 我们正在向结果接收器流式传输
                sink = new ResultSink(planRoot.getId(), resultSinkType);
            }
        }
    }

    /**
     * 返回计划片段将执行的节点数量。
     * 无效：-1
     */
    public int getNumNodes() {
        return dataPartition == DataPartition.UNPARTITIONED ? 1 : planRoot.getNumNodes();
    }

    public int getParallelExecNum() {
        return parallelExecNum;
    }

    public TPlanFragment toThrift() {
        TPlanFragment result = new TPlanFragment();
        if (planRoot != null) {
            result.setPlan(planRoot.treeToThrift());
        }
        if (outputExprs != null) {
            result.setOutputExprs(Expr.treesToThrift(outputExprs));
        }
        if (sink != null) {
            result.setOutputSink(sink.toThrift());
        }
        if (dataPartitionForThrift == null) {
            result.setPartition(dataPartition.toThrift());
        } else {
            result.setPartition(dataPartitionForThrift.toThrift());
        }

        // TODO chenhao , 根据成本计算
        result.setMinReservationBytes(0);
        result.setInitialReservationTotalClaims(0);
        return result;
    }

    public String getExplainString(TExplainLevel explainLevel) {
        StringBuilder str = new StringBuilder();
        Preconditions.checkState(dataPartition != null);
        if (CollectionUtils.isNotEmpty(outputExprs)) {
            str.append("  OUTPUT EXPRS:\n    ");
            str.append(outputExprs.stream().map(Expr::toSql).collect(Collectors.joining("\n    ")));
        }
        str.append("\n");
        str.append("  PARTITION: " + dataPartition.getExplainString(explainLevel) + "\n");
        str.append("  HAS_COLO_PLAN_NODE: " + hasColocatePlanNode + "\n");
        str.append("\n");
        if (sink != null) {
            str.append(sink.getExplainString("  ", explainLevel) + "\n");
        }
        if (planRoot != null) {
            str.append(planRoot.getExplainString("  ", "  ", explainLevel));
        }
        return str.toString();
    }

    public void getExplainStringMap(Map<Integer, String> planNodeMap) {
        org.apache.doris.thrift.TExplainLevel explainLevel = org.apache.doris.thrift.TExplainLevel.NORMAL;
        if (planRoot != null) {
            planRoot.getExplainStringMap(explainLevel, planNodeMap);
        }
    }

    /**
     * 如果此片段是分区的，则返回true。
     */
    public boolean isPartitioned() {
        return (dataPartition.getType() != TPartitionType.UNPARTITIONED);
    }

    public void updateDataPartition(DataPartition dataPartition) {
        if (this.dataPartition == DataPartition.UNPARTITIONED) {
            return;
        }
        this.dataPartition = dataPartition;
    }

    public PlanFragmentId getId() {
        return fragmentId;
    }

    public PlanFragment getDestFragment() {
        if (destNode == null) {
            return null;
        }
        return destNode.getFragment();
    }

    public void setDestination(ExchangeNode destNode) {
        this.destNode = destNode;
        PlanFragment dest = getDestFragment();
        Preconditions.checkNotNull(dest);
        dest.addChild(this);
    }

    public DataPartition getDataPartition() {
        return dataPartition;
    }

    public DataPartition getOutputPartition() {
        return outputPartition;
    }

    public void setOutputPartition(DataPartition outputPartition) {
        this.outputPartition = outputPartition;
    }

    public PlanNode getPlanRoot() {
        return planRoot;
    }

    public void setPlanRoot(PlanNode root) {
        planRoot = root;
        setFragmentInPlanTree(planRoot);
    }

    /**
     * 将一个节点添加为计划树的新根节点。将现有的根节点连接为newRoot的子节点。
     */
    public void addPlanRoot(PlanNode newRoot) {
        Preconditions.checkState(newRoot.getChildren().size() == 1);
        newRoot.setChild(0, planRoot);
        planRoot = newRoot;
        planRoot.setFragment(this);
    }

    public DataSink getSink() {
        return sink;
    }

    public void setSink(DataSink sink) {
        Preconditions.checkState(this.sink == null);
        Preconditions.checkNotNull(sink);
        sink.setFragment(this);
        this.sink = sink;
    }

    public void resetSink(DataSink sink) {
        sink.setFragment(this);
        this.sink = sink;
    }

    public PlanFragmentId getFragmentId() {
        return fragmentId;
    }

    public Set<RuntimeFilterId> getBuilderRuntimeFilterIds() {
        return builderRuntimeFilterIds;
    }

    public Set<RuntimeFilterId> getTargetRuntimeFilterIds() {
        return targetRuntimeFilterIds;
    }

    public void clearRuntimeFilters() {
        builderRuntimeFilterIds.clear();
        targetRuntimeFilterIds.clear();
    }

    public void setTransferQueryStatisticsWithEveryBatch(boolean value) {
        transferQueryStatisticsWithEveryBatch = value;
    }

    public boolean isTransferQueryStatisticsWithEveryBatch() {
        return transferQueryStatisticsWithEveryBatch;
    }

    public int getFragmentSequenceNum() {
        if (ConnectContext.get().getSessionVariable().isEnableNereidsPlanner()) {
            return fragmentSequenceNum;
        } else {
            return fragmentId.asInt();
        }
    }

    public void setFragmentSequenceNum(int seq) {
        fragmentSequenceNum = seq;
    }

    public int getBucketNum() {
        return bucketNum;
    }

    public void setBucketNum(int bucketNum) {
        this.bucketNum = bucketNum;
    }

    public boolean hasNullAwareLeftAntiJoin() {
        return planRoot.isNullAwareLeftAntiJoin();
    }
}
