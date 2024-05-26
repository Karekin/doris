// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.
// This file is copied from
// https://github.com/apache/impala/blob/branch-2.9.0/fe/src/main/java/org/apache/impala/PlanNode.java
// and modified by Doris

package org.apache.doris.planner;

import org.apache.doris.analysis.Analyzer;
import org.apache.doris.analysis.BitmapFilterPredicate;
import org.apache.doris.analysis.CompoundPredicate;
import org.apache.doris.analysis.Expr;
import org.apache.doris.analysis.ExprId;
import org.apache.doris.analysis.ExprSubstitutionMap;
import org.apache.doris.analysis.FunctionCallExpr;
import org.apache.doris.analysis.SlotId;
import org.apache.doris.analysis.SlotRef;
import org.apache.doris.analysis.TupleDescriptor;
import org.apache.doris.analysis.TupleId;
import org.apache.doris.catalog.Column;
import org.apache.doris.catalog.OlapTable;
import org.apache.doris.common.AnalysisException;
import org.apache.doris.common.NotImplementedException;
import org.apache.doris.common.TreeNode;
import org.apache.doris.common.UserException;
import org.apache.doris.statistics.PlanStats;
import org.apache.doris.statistics.StatisticalType;
import org.apache.doris.statistics.StatsDeriveResult;
import org.apache.doris.thrift.TExplainLevel;
import org.apache.doris.thrift.TPlan;
import org.apache.doris.thrift.TPlanNode;
import org.apache.doris.thrift.TPushAggOp;

import com.google.common.base.Joiner;
import com.google.common.base.Preconditions;
import com.google.common.base.Predicates;
import com.google.common.collect.Lists;
import com.google.common.collect.Sets;
import org.apache.commons.collections.CollectionUtils;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * 每个PlanNode表示一个单独的关系运算符，并封装了规划器在进行优化决策时所需的信息。

 * finalize(): 计算内部状态，如扫描节点的键；在调用toThrift()之前，根计划树上的此方法只调用一次。
 * 还会完成连接子集的设置，使得每个剩余的连接子集都需要它所引用的所有插槽物化（即可以通过调用GetValue()来评估，
 * 而不是作为扫描键的一部分隐式评估）。

 * conjuncts: 每个节点都有一个连接子集列表，这些子集可以在该节点的上下文中执行，即，它们仅引用由该节点或其子节点物化的元组（=受元组ID绑定）。
 */
public abstract class PlanNode extends TreeNode<PlanNode> implements PlanStats {

    protected String planNodeName; // 计划节点的名称

    protected PlanNodeId id;  // 在计划树中唯一，由规划器分配
    protected PlanFragmentId fragmentId;  // 在分段步骤后由规划器分配
    protected long limit; // 返回的最大行数；0表示无限制
    protected long offset; // 偏移量

    /*
     * tupleIds 字段表示在当前计划节点（PlanNode）所对应的计划树中，被物化（materialized）的元组（Tuple）的ID列表。
     * 换句话说，这个列表包含了该节点及其子节点在执行过程中实际生成和使用的元组ID，这些元组是查询执行过程中需要被具体化并存储的。
     * 可以形象地将 TupleId 看作是每个数据行（Row）的身份证号，用于标识和引用查询过程中生成或处理的具体数据行。
     */
    protected ArrayList<TupleId> tupleIds;

    /*
     * `tblRefIds` 字段表示在当前计划节点中被“物化”的表引用（Table References）的ID列表；
     * 如果当前节点及其子节点的计划树仅“物化”基本表引用（BaseTblRefs），则这个列表与 `tupleIds` 相同；这个字段在生成查询执行计划时非常有用。

     * tblRefIds 字段表示当前计划节点及其子节点所涉及到的表引用的ID列表。如果这个计划节点只涉及到基本表（例如物理存在的数据库表），
     * 那么这个字段与 tupleIds 是相同的。如果计划节点涉及到的是其他类型的表引用（例如视图、派生表或临时表），那么这个字段将包含这些特殊表引用的ID。

     * 两者的区别在于：
     * 1. 基本表（BaseTblRefs）：这是数据库中实际存在的物理表，直接存储数据。
     *    计划节点涉及到基本表时， tblRefIds 和 tupleIds 是相同的，因为所有元组都来自这些基本表。
     * 2. 其他表引用：这些可以是视图（由查询定义的虚拟表）、派生表（查询中产生的临时表）或其他临时表。
     *    这些表引用不直接存储数据，而是通过查询生成的数据集。计划节点涉及到这些表引用时， tblRefIds 可能不同于 tupleIds ，
     *    因为这些表引用需要额外的查询处理步骤来生成元组。
     */
    protected ArrayList<TupleId> tblRefIds;

    // 由此节点生成的可为空的TupleId集合。这是tupleIds的子集。
    // 元组在特定计划树中是可空的，如果它是外连接的“可空”端，这与模式无关。
    protected Set<TupleId> nullableTupleIds = Sets.newHashSet();

    protected List<Expr> conjuncts = Lists.newArrayList(); // 连接子集列表


    /*
       用于过滤原始加载文件的连接子集。
       在加载执行计划中，“preFilterConjuncts”和“conjuncts”的区别在于
       conjuncts用于在列转换和映射后过滤数据，而fileFilterConjuncts直接过滤从源数据读取的内容。
       即，数据处理流程如下：
        1. 从源读取数据。
        2. 使用“preFilterConjuncts”过滤数据。
        3. 进行列映射和转换。
        4. 使用“conjuncts”过滤数据。

       列转换是对数据列进行转换操作，使其符合目标表的要求。
       列映射是将源数据的列与目标表的列进行匹配或对应。
       preFilterConjuncts 用于在列转换和映射之前对原始数据进行初步过滤，减少后续处理负担。
       conjuncts 用于在列转换和映射之后对数据进行最终过滤，确保数据符合加载或查询要求。
     */
    protected List<Expr> preFilterConjuncts = Lists.newArrayList();

    protected Expr vpreFilterConjunct = null;

    // 此PlanNode在执行的片段。仅在此PlanNode已分配给片段后有效。
    // 由包含的PlanFragment设置和维护。
    protected PlanFragment fragment;

    // 此节点的输出基数估计值；在computeStats()中设置；
    // 无效：-1
    protected long cardinality;

    protected long cardinalityAfterFilter = -1; // 过滤后的基数

    // 此节点根的计划树在其上执行的节点数；
    // 在computeStats()中设置；无效：-1
    protected int numNodes;

    // tupleIds的avgSerializedSizes总和；在computeStats()中设置
    protected float avgRowSize;

    // 节点应压缩数据。
    protected boolean compactData;

    // 大多数计划节点的numInstance与其（左）子节点相同，除非一些特殊节点，如：
    // 1. 扫描节点，其numInstance根据其数据分布计算
    // 2. 交换节点，这是汇聚分布
    // 3. 联合节点，其numInstance是其子节点的numInstance之和
    // ...
    // 只有特殊节点需要调用setNumInstances()和getNumInstances()从属性numInstances获取
    protected int numInstances;

    // 分配给此节点的运行时过滤器。
    protected List<RuntimeFilter> runtimeFilters = new ArrayList<>();

    protected List<SlotId> outputSlotIds; // 输出槽ID列表

    protected StatisticalType statisticalType = StatisticalType.DEFAULT; // 统计类型
    protected StatsDeriveResult statsDeriveResult; // 统计派生结果

    protected TupleDescriptor outputTupleDesc; // 输出元组描述


    protected List<Expr> projectList; // 投影列表

    protected int nereidsId = -1; // nereidsId

    private List<List<Expr>> childrenDistributeExprLists = new ArrayList<>(); // 子节点分布表达式列表
    private List<TupleDescriptor> intermediateOutputTupleDescList = Lists.newArrayList(); // 中间输出元组描述列表
    private List<List<Expr>> intermediateProjectListList = Lists.newArrayList(); // 中间投影列表

    protected PlanNode(PlanNodeId id, ArrayList<TupleId> tupleIds, String planNodeName, StatisticalType statisticalType) {
        this.id = id;
        this.limit = -1;
        this.offset = 0;
        // 复制一份，以防万一
        this.tupleIds = Lists.newArrayList(tupleIds);
        this.tblRefIds = Lists.newArrayList(tupleIds);
        this.cardinality = -1;
        this.planNodeName = "V" + planNodeName;
        this.numInstances = 1;
        this.statisticalType = statisticalType;
    }

    protected PlanNode(PlanNodeId id, String planNodeName, StatisticalType statisticalType) {
        this.id = id;
        this.limit = -1;
        this.tupleIds = Lists.newArrayList();
        this.tblRefIds = Lists.newArrayList();
        this.cardinality = -1;
        this.planNodeName = "V" + planNodeName;
        this.numInstances = 1;
        this.statisticalType = statisticalType;
    }

    /**
     * 复制构造函数。也传递新id。
     */
    protected PlanNode(PlanNodeId id, PlanNode node, String planNodeName, StatisticalType statisticalType) {
        this.id = id;
        this.limit = node.limit;
        this.offset = node.offset;
        this.tupleIds = Lists.newArrayList(node.tupleIds);
        this.tblRefIds = Lists.newArrayList(node.tblRefIds);
        this.nullableTupleIds = Sets.newHashSet(node.nullableTupleIds);
        this.conjuncts = Expr.cloneList(node.conjuncts, null);
        this.cardinality = -1;
        this.compactData = node.compactData;
        this.planNodeName = "V" + planNodeName;
        this.numInstances = 1;
        this.statisticalType = statisticalType;
    }

    public String getPlanNodeName() {
        return planNodeName;
    }

    public StatsDeriveResult getStatsDeriveResult() {
        return statsDeriveResult;
    }

    public StatisticalType getStatisticalType() {
        return statisticalType;
    }

    public void setStatsDeriveResult(StatsDeriveResult statsDeriveResult) {
        this.statsDeriveResult = statsDeriveResult;
    }

    /**
     * 设置tblRefIds_、tupleIds_和nullableTupleIds_。
     * 默认实现不执行任何操作。
     */
    public void computeTupleIds() {
        Preconditions.checkState(children.isEmpty() || !tupleIds.isEmpty());
    }

    /**
     * 清除tblRefIds_、tupleIds_和nullableTupleIds_。
     */
    protected void clearTupleIds() {
        tblRefIds.clear();
        tupleIds.clear();
        nullableTupleIds.clear();
    }

    protected void setPlanNodeName(String s) {
        this.planNodeName = s;
    }

    public PlanNodeId getId() {
        return id;
    }

    public void setId(PlanNodeId id) {
        Preconditions.checkState(this.id == null);
        this.id = id;
    }

    public PlanFragmentId getFragmentId() {
        return fragment.getFragmentId();
    }

    public int getFragmentSeqenceNum() {
        return fragment.getFragmentSequenceNum();
    }

    public void setFragmentId(PlanFragmentId id) {
        fragmentId = id;
    }

    public void setFragment(PlanFragment fragment) {
        this.fragment = fragment;
    }

    public boolean isNullAwareLeftAntiJoin() {
        return children.stream().anyMatch(PlanNode::isNullAwareLeftAntiJoin);
    }

    public PlanFragment getFragment() {
        return fragment;
    }

    public long getLimit() {
        return limit;
    }

    public long getOffset() {
        return offset;
    }

    /**
     * 仅在未设置限制或新限制较低时，将限制设置为给定限制。
     *
     * @param limit 新的限制值
     */
    public void setLimit(long limit) {
        if (this.limit == -1 || (limit != -1 && this.limit > limit)) {
            this.limit = limit;
        }
    }

    public void setLimitAndOffset(long limit, long offset) {
        if (this.limit == -1) {
            this.limit = limit;
        } else if (limit != -1) {
            this.limit = Math.min(this.limit - offset, limit);
        }
        this.offset += offset;
    }

    public void setOffset(long offset) {
        this.offset = offset;
    }

    /**
     * 仅供新优化器使用。
     */
    public void setOffSetDirectly(long offset) {
        this.offset = offset;
    }

    public boolean hasLimit() {
        return limit > -1;
    }

    public boolean hasOffset() {
        return offset != 0;
    }

    public void setCardinality(long cardinality) {
        this.cardinality = cardinality;
    }

    public long getCardinality() {
        return cardinality;
    }

    public long getCardinalityAfterFilter() {
        if (cardinalityAfterFilter < 0) {
            return cardinality;
        } else {
            return cardinalityAfterFilter;
        }
    }

    public int getNumNodes() {
        return numNodes;
    }

    public float getAvgRowSize() {
        return avgRowSize;
    }

    /**
     * 设置所有子节点的compactData值。
     */
    public void setCompactData(boolean on) {
        this.compactData = on;
        for (PlanNode child : this.getChildren()) {
            child.setCompactData(on);
        }
    }

    public void unsetLimit() {
        limit = -1;
    }

    protected List<TupleId> getAllScanTupleIds() {
        List<TupleId> tupleIds = Lists.newArrayList();
        List<ScanNode> scanNodes = Lists.newArrayList();
        collectAll(Predicates.instanceOf(ScanNode.class), scanNodes);
        for (ScanNode node : scanNodes) {
            tupleIds.addAll(node.getTupleIds());
        }
        return tupleIds;
    }

    public void resetTupleIds(ArrayList<TupleId> tupleIds) {
        this.tupleIds = tupleIds;
    }

    public ArrayList<TupleId> getTupleIds() {
        Preconditions.checkState(tupleIds != null);
        return tupleIds;
    }

    public ArrayList<TupleId> getTblRefIds() {
        return tblRefIds;
    }

    public void setTblRefIds(ArrayList<TupleId> ids) {
        tblRefIds = ids;
    }

    public ArrayList<TupleId> getOutputTblRefIds() {
        return tblRefIds;
    }

    public List<TupleId> getOutputTupleIds() {
        if (outputTupleDesc != null) {
            return Lists.newArrayList(outputTupleDesc.getId());
        }
        return tupleIds;
    }

    public Set<TupleId> getNullableTupleIds() {
        Preconditions.checkState(nullableTupleIds != null);
        return nullableTupleIds;
    }

    public List<Expr> getConjuncts() {
        return conjuncts;
    }

    @Override
    public List<StatsDeriveResult> getChildrenStats() {
        List<StatsDeriveResult> statsDeriveResultList = Lists.newArrayList();
        for (PlanNode child : children) {
            statsDeriveResultList.add(child.getStatsDeriveResult());
        }
        return statsDeriveResultList;
    }

    public static Expr convertConjunctsToAndCompoundPredicate(List<Expr> conjuncts) {
        List<Expr> targetConjuncts = Lists.newArrayList(conjuncts);
        while (targetConjuncts.size() > 1) {
            List<Expr> newTargetConjuncts = Lists.newArrayList();
            for (int i = 0; i < targetConjuncts.size(); i += 2) {
                Expr expr = i + 1 < targetConjuncts.size()
                    ? new CompoundPredicate(CompoundPredicate.Operator.AND, targetConjuncts.get(i),
                    targetConjuncts.get(i + 1)) : targetConjuncts.get(i);
                newTargetConjuncts.add(expr);
            }
            targetConjuncts = newTargetConjuncts;
        }

        Preconditions.checkArgument(targetConjuncts.size() == 1);
        return targetConjuncts.get(0);
    }

    public static List<Expr> splitAndCompoundPredicateToConjuncts(Expr vconjunct) {
        List<Expr> conjuncts = Lists.newArrayList();
        if (vconjunct instanceof CompoundPredicate) {
            CompoundPredicate andCompound = (CompoundPredicate) vconjunct;
            if (andCompound.getOp().equals(CompoundPredicate.Operator.AND)) {
                conjuncts.addAll(splitAndCompoundPredicateToConjuncts(vconjunct.getChild(0)));
                conjuncts.addAll(splitAndCompoundPredicateToConjuncts(vconjunct.getChild(1)));
            }
        }
        if (vconjunct != null && conjuncts.isEmpty()) {
            conjuncts.add(vconjunct);
        }
        return conjuncts;
    }

    public void addConjuncts(List<Expr> conjuncts) {
        if (conjuncts == null) {
            return;
        }
        for (Expr conjunct : conjuncts) {
            addConjunct(conjunct);
        }
    }

    public void addConjunct(Expr conjunct) {
        if (conjuncts == null) {
            conjuncts = Lists.newArrayList();
        }
        if (!conjuncts.contains(conjunct)) {
            conjuncts.add(conjunct);
        }
    }

    public void setAssignedConjuncts(Set<ExprId> conjuncts) {
        assignedConjuncts = conjuncts;
    }

    public Set<ExprId> getAssignedConjuncts() {
        return assignedConjuncts;
    }

    public void transferConjuncts(PlanNode recipient) {
        recipient.conjuncts.addAll(conjuncts);
        conjuncts.clear();
    }

    public void addPreFilterConjuncts(List<Expr> conjuncts) {
        if (conjuncts == null) {
            return;
        }
        this.preFilterConjuncts.addAll(conjuncts);
    }

    /**
     * 调用computeStatAndMemLayout()以获取所有物化的元组。
     */
    protected void computeTupleStatAndMemLayout(Analyzer analyzer) {
        for (TupleId id : tupleIds) {
            analyzer.getDescTbl().getTupleDesc(id).computeStatAndMemLayout();
        }
    }

    public String getExplainString() {
        return getExplainString("", "", TExplainLevel.VERBOSE);
    }

    /**
     * 生成解释计划树。计划的形式如下：
     *
     * root
     * |
     * |----child 2
     * |      limit:1
     * |
     * |----child 3
     * |      limit:2
     * |
     * child 1
     *
     * 根节点头行将由rootPrefix前缀，其余计划输出将由prefix前缀。
     */
    protected final String getExplainString(String rootPrefix, String prefix, TExplainLevel detailLevel) {
        StringBuilder expBuilder = new StringBuilder();
        String detailPrefix = prefix;
        boolean traverseChildren = children != null && children.size() > 0 && !(this instanceof ExchangeNode);
        if (traverseChildren) {
            detailPrefix += "|  ";
        } else {
            detailPrefix += "   ";
        }

        // Print the current node
        // The plan node header line will be prefixed by rootPrefix and the remaining details
        // will be prefixed by detailPrefix.
        expBuilder.append(rootPrefix + id.asInt() + ":" + planNodeName);
        if (nereidsId != -1) {
            expBuilder.append("(" + nereidsId + ")");
        }
        expBuilder.append("\n");
        expBuilder.append(getNodeExplainString(detailPrefix, detailLevel));
        if (limit != -1) {
            expBuilder.append(detailPrefix + "limit: " + limit + "\n");
        }
        if (!CollectionUtils.isEmpty(projectList)) {
            expBuilder.append(detailPrefix).append("final projections: ")
                .append(getExplainString(projectList)).append("\n");
            expBuilder.append(detailPrefix).append("final project output tuple id: ")
                .append(outputTupleDesc.getId().asInt()).append("\n");
        }
        if (!intermediateProjectListList.isEmpty()) {
            int layers = intermediateProjectListList.size();
            for (int i = layers - 1; i >= 0; i--) {
                expBuilder.append(detailPrefix).append("intermediate projections: ")
                    .append(getExplainString(intermediateProjectListList.get(i))).append("\n");
                expBuilder.append(detailPrefix).append("intermediate tuple id: ")
                    .append(intermediateOutputTupleDescList.get(i).getId().asInt()).append("\n");
            }
        }
        if (!CollectionUtils.isEmpty(childrenDistributeExprLists)) {
            for (List<Expr> distributeExprList : childrenDistributeExprLists) {
                expBuilder.append(detailPrefix).append("distribute expr lists: ")
                    .append(getExplainString(distributeExprList)).append("\n");
            }
        }
        // 仅在解释计划级别设置为详细时输出元组ID
        if (detailLevel.equals(TExplainLevel.VERBOSE)) {
            expBuilder.append(detailPrefix + "tuple ids: ");
            for (TupleId tupleId : tupleIds) {
                String nullIndicator = nullableTupleIds.contains(tupleId) ? "N" : "";
                expBuilder.append(tupleId.asInt() + nullIndicator + " ");
            }
            expBuilder.append("\n");
        }

        // 打印子节点
        if (traverseChildren) {
            expBuilder.append(detailPrefix + "\n");
            String childHeadlinePrefix = prefix + "|----";
            String childDetailPrefix = prefix + "|    ";
            for (int i = 1; i < children.size(); ++i) {
                expBuilder.append(
                    children.get(i).getExplainString(childHeadlinePrefix, childDetailPrefix,
                        detailLevel));
                expBuilder.append(childDetailPrefix + "\n");
            }
            expBuilder.append(children.get(0).getExplainString(prefix, prefix, detailLevel));
        }
        return expBuilder.toString();
    }

    private String getplanNodeExplainString(String prefix, TExplainLevel detailLevel) {
        StringBuilder expBuilder = new StringBuilder();
        expBuilder.append(getNodeExplainString(prefix, detailLevel));
        if (limit != -1) {
            expBuilder.append(prefix + "limit: " + limit + "\n");
        }
        if (!CollectionUtils.isEmpty(projectList)) {
            expBuilder.append(prefix).append("projections: ").append(getExplainString(projectList)).append("\n");
            expBuilder.append(prefix).append("project output tuple id: ")
                .append(outputTupleDesc.getId().asInt()).append("\n");
        }
        return expBuilder.toString();
    }

    public void getExplainStringMap(TExplainLevel detailLevel, Map<Integer, String> planNodeMap) {
        planNodeMap.put(id.asInt(), getplanNodeExplainString("", detailLevel));
        for (int i = 0; i < children.size(); ++i) {
            children.get(i).getExplainStringMap(detailLevel, planNodeMap);
        }
    }

    /**
     * 返回节点特定的详细信息。
     * 子类应覆盖此方法。
     * 每行应以detailPrefix为前缀。
     */
    public String getNodeExplainString(String prefix, TExplainLevel detailLevel) {
        return "";
    }

    // 将此计划节点（包括所有子节点）转换为其Thrift表示。
    public TPlan treeToThrift() {
        TPlan result = new TPlan();
        treeToThriftHelper(result);
        return result;
    }

    // 将此计划节点（包括所有子节点）的扁平版本附加到'container'。
    private void treeToThriftHelper(TPlan container) {
        TPlanNode msg = new TPlanNode();
        msg.node_id = id.asInt();
        msg.num_children = children.size();
        msg.limit = limit;
        for (TupleId tid : tupleIds) {
            msg.addToRowTuples(tid.asInt());
            msg.addToNullableTuples(nullableTupleIds.contains(tid));
        }

        for (Expr e : conjuncts) {
            if  (!(e instanceof BitmapFilterPredicate)) {
                msg.addToConjuncts(e.treeToThrift());
            }
        }

        // 序列化任何运行时过滤器
        for (RuntimeFilter filter : runtimeFilters) {
            msg.addToRuntimeFilters(filter.toThrift());
        }

        msg.compact_data = compactData;
        if (outputSlotIds != null) {
            for (SlotId slotId : outputSlotIds) {
                msg.addToOutputSlotIds(slotId.asInt());
            }
        }
        if (!CollectionUtils.isEmpty(childrenDistributeExprLists)) {
            for (List<Expr> exprList : childrenDistributeExprLists) {
                msg.addToDistributeExprLists(new ArrayList<>());
                for (Expr expr : exprList) {
                    msg.distribute_expr_lists.get(msg.distribute_expr_lists.size() - 1).add(expr.treeToThrift());
                }
            }
        }
        toThrift(msg);
        container.addToNodes(msg);

        // 旧规划器在连接节点内设置输出元组和投影
        if (!(this instanceof JoinNodeBase) || !(((JoinNodeBase) this).isUseSpecificProjections())) {
            if (outputTupleDesc != null) {
                msg.setOutputTupleId(outputTupleDesc.getId().asInt());
            }
            if (projectList != null) {
                for (Expr expr : projectList) {
                    msg.addToProjections(expr.treeToThrift());
                }
            }
        }

        if (!intermediateOutputTupleDescList.isEmpty()) {
            intermediateOutputTupleDescList
                .forEach(
                    tupleDescriptor -> msg.addToIntermediateOutputTupleIdList(tupleDescriptor.getId().asInt()));
        }

        if (!intermediateProjectListList.isEmpty()) {
            intermediateProjectListList.forEach(
                projectList -> msg.addToIntermediateProjectionsList(
                    projectList.stream().map(expr -> expr.treeToThrift()).collect(Collectors.toList())));
        }

        if (this instanceof ExchangeNode) {
            msg.num_children = 0;
            return;
        } else {
            msg.num_children = children.size();
            for (PlanNode child : children) {
                child.treeToThriftHelper(container);
            }
        }
    }

    /**
     * 计算内部状态，包括规划器相关统计信息。
     * 在调用toThrift()之前，在计划树的根上调用此方法一次。
     * 子类需要覆盖此方法。
     */
    public void finalize(Analyzer analyzer) throws UserException {
        for (Expr expr : conjuncts) {
            Set<SlotRef> slotRefs = new HashSet<>();
            expr.getSlotRefsBoundByTupleIds(tupleIds, slotRefs);
            for (SlotRef slotRef : slotRefs) {
                slotRef.getDesc().setIsMaterialized(true);
            }
            for (TupleId tupleId : tupleIds) {
                analyzer.getTupleDesc(tupleId).computeMemLayout();
            }
        }
        for (PlanNode child : children) {
            child.finalize(analyzer);
        }
        computeNumNodes();
        if (!analyzer.safeIsEnableJoinReorderBasedCost()) {
            computeOldCardinality();
        }
    }

    protected void computeNumNodes() {
        if (!children.isEmpty()) {
            numNodes = getChild(0).numNodes;
        }
    }

    /**
     * 计算规划器统计信息：avgRowSize。
     * 子类需要覆盖此方法。
     * 假设已经在所有子节点上调用了此方法。
     * 这是从finalize()中分离出来的，因此可以单独调用（以便在计划分区期间插入其他节点，而无需再次递归地调用整个树的finalize()）。
     */
    protected void computeStats(Analyzer analyzer) throws UserException {
        avgRowSize = 0.0F;
        for (TupleId tid : tupleIds) {
            TupleDescriptor desc = analyzer.getTupleDesc(tid);
            avgRowSize += desc.getAvgSerializedSize();
        }
    }

    /**
     * 此函数在启用旧连接重排序算法时计算基数。
     * 此值用于在分布式规划中确定连接的分布方式（广播或洗牌）。
     *
     * 如果新连接重排序和旧连接重排序具有相同的基数计算方法，并且计算已在init()中完成，
     * 则无需覆盖此函数。
     */
    protected void computeOldCardinality() {
    }

    protected void capCardinalityAtLimit() {
        if (hasLimit()) {
            cardinality = cardinality == -1 ? limit : Math.min(cardinality, limit);
        }
    }

    protected ExprSubstitutionMap outputSmap;

    // 关于连接子集分配的全局状态；由规划器用作快捷方式，避免在连接树替代之间来回传递分配的连接子集
    // （规划器使用此功能保存和重置全局状态）。
    protected Set<ExprId> assignedConjuncts;

    protected ExprSubstitutionMap withoutTupleIsNullOutputSmap;

    public ExprSubstitutionMap getOutputSmap() {
        return outputSmap;
    }

    public void setOutputSmap(ExprSubstitutionMap smap, Analyzer analyzer) {
        outputSmap = smap;
    }

    public void setWithoutTupleIsNullOutputSmap(ExprSubstitutionMap smap) {
        withoutTupleIsNullOutputSmap = smap;
    }

    public ExprSubstitutionMap getWithoutTupleIsNullOutputSmap() {
        return withoutTupleIsNullOutputSmap == null ? outputSmap : withoutTupleIsNullOutputSmap;
    }

    public void init() throws UserException {}

    public void init(Analyzer analyzer) throws UserException {
        assignConjuncts(analyzer);
        createDefaultSmap(analyzer);
    }

    /**
     * 分配剩余的未分配连接子集。
     */
    protected void assignConjuncts(Analyzer analyzer) {
        if (this instanceof ExchangeNode) {
            return;
        }
        List<Expr> unassigned = analyzer.getUnassignedConjuncts(this);
        for (Expr unassignedConjunct : unassigned) {
            addConjunct(unassignedConjunct);
        }
        analyzer.markConjunctsAssigned(unassigned);
    }

    /**
     * 返回组合子节点smap的结果。
     */
    protected ExprSubstitutionMap getCombinedChildSmap() {
        if (getChildren().size() == 0) {
            return new ExprSubstitutionMap();
        }

        if (getChildren().size() == 1) {
            return getChild(0).getOutputSmap();
        }

        ExprSubstitutionMap result = ExprSubstitutionMap.combine(
            getChild(0).getOutputSmap(), getChild(1).getOutputSmap());

        for (int i = 2; i < getChildren().size(); ++i) {
            result = ExprSubstitutionMap.combine(result, getChild(i).getOutputSmap());
        }

        return result;
    }

    protected ExprSubstitutionMap getCombinedChildWithoutTupleIsNullSmap() {
        if (getChildren().size() == 0) {
            return new ExprSubstitutionMap();
        }
        if (getChildren().size() == 1) {
            return getChild(0).getWithoutTupleIsNullOutputSmap();
        }
        ExprSubstitutionMap result = ExprSubstitutionMap.combine(
            getChild(0).getWithoutTupleIsNullOutputSmap(),
            getChild(1).getWithoutTupleIsNullOutputSmap());

        for (int i = 2; i < getChildren().size(); ++i) {
            result = ExprSubstitutionMap.combine(
                result, getChild(i).getWithoutTupleIsNullOutputSmap());
        }

        return result;
    }

    /**
     * 将outputSmap_设置为compose(existing smap, combined child smap)。还
     * 使用组合的子节点smap替换conjuncts_。
     *
     * @throws AnalysisException
     */
    protected void createDefaultSmap(Analyzer analyzer) throws UserException {
        ExprSubstitutionMap combinedChildSmap = getCombinedChildSmap();
        outputSmap =
            ExprSubstitutionMap.compose(outputSmap, combinedChildSmap, analyzer);

        conjuncts = Expr.substituteList(conjuncts, outputSmap, analyzer, false);
    }

    /**
     * 添加需要物化的槽ID以获取此节点树。
     * 默认情况下，仅引用的槽需要物化
     * （其理由是仅需要显式评估连接子集；
     * 变成扫描谓词等的表达式是隐式评估的）。
     */
    public void getMaterializedIds(Analyzer analyzer, List<SlotId> ids) {
        for (PlanNode childNode : children) {
            childNode.getMaterializedIds(analyzer, ids);
        }
        Expr.getIds(getConjuncts(), null, ids);
    }

    // 将此计划节点（不包括子节点）转换为msg，需要设置节点类型和节点特定字段。
    protected abstract void toThrift(TPlanNode msg);

    protected String debugString() {
        StringBuilder output = new StringBuilder();
        output.append("preds=" + Expr.debugString(conjuncts));
        output.append(" limit=" + Long.toString(limit));
        return output.toString();
    }

    public static String getExplainString(List<? extends Expr> exprs) {
        if (exprs == null) {
            return "";
        }
        StringBuilder output = new StringBuilder();
        for (int i = 0; i < exprs.size(); ++i) {
            if (i > 0) {
                output.append(", ");
            }
            output.append(exprs.get(i).toSql());
        }
        return output.toString();
    }

    /**
     * 返回true，如果统计相关变量有效。
     */
    protected boolean hasValidStats() {
        return (numNodes == -1 || numNodes >= 0) && (cardinality == -1 || cardinality >= 0);
    }

    public int getNumInstances() {
        return this.children.get(0).getNumInstances();
    }

    public void setShouldColoScan() {}

    public boolean getShouldColoScan() {
        return false;
    }

    public void setNumInstances(int numInstances) {
        this.numInstances = numInstances;
    }

    public void appendTrace(StringBuilder sb) {
        sb.append(planNodeName);
        if (!children.isEmpty()) {
            sb.append("(");
            int idx = 0;
            for (PlanNode child : children) {
                if (idx++ != 0) {
                    sb.append(",");
                }
                child.appendTrace(sb);
            }
            sb.append(")");
        }
    }

    /**
     * 返回所有连接子集的估计组合选择性。使用启发式方法解决以下估计挑战：
     * 1. 连接子集的个别选择性可能是未知的。
     * 2. 两个选择性，无论是已知还是未知，都可能是相关的。假设独立性可能导致严重的低估。
     *
     * 第一个问题通过使用单个默认选择性来解决，该选择性代表所有未知选择性的连接子集。
     * 第二个问题通过在最终结果中乘以每个附加选择性时的指数回退来解决。
     */
    protected static double computeCombinedSelectivity(List<Expr> conjuncts) {
        // 收集所有估计的选择性。
        List<Double> selectivities = new ArrayList<>();
        for (Expr e : conjuncts) {
            if (e.hasSelectivity()) {
                selectivities.add(e.getSelectivity());
            }
        }
        if (selectivities.size() != conjuncts.size()) {
            // 某些连接子集没有估计的选择性。使用单个默认代表选择性表示所有这些连接子集。
            selectivities.add(Expr.DEFAULT_SELECTIVITY);
        }
        // 对选择性进行排序以获得一致的估计，不论原始连接子集顺序如何。按升序排序，使得最具选择性的连接子集被完全应用。
        Collections.sort(selectivities);
        double result = 1.0;
        // selectivity = 1 * (s1)^(1/1) * (s2)^(1/2) * ... * (sn-1)^(1/(n-1)) * (sn)^(1/n)
        for (int i = 0; i < selectivities.size(); ++i) {
            // 乘入最终结果中的每个选择性的指数回退。
            result *= Math.pow(selectivities.get(i), 1.0 / (double) (i + 1));
        }
        // 将结果限制在[0, 1]范围内
        return Math.max(0.0, Math.min(1.0, result));
    }

    protected double computeSelectivity() {
        for (Expr expr : conjuncts) {
            expr.setSelectivity();
        }
        return computeCombinedSelectivity(conjuncts);
    }

    /**
     * 计算所有连接子集的选择性的乘积。
     * 此函数用于finalize()中的旧基数
     */
    protected double computeOldSelectivity() {
        double prod = 1.0;
        for (Expr e : conjuncts) {
            if (e.getSelectivity() < 0) {
                return -1.0;
            }
            prod *= e.getSelectivity();
        }
        return prod;
    }

    // 根据'preConjunctCardinality'计算应用连接子集后的基数。
    protected void applyConjunctsSelectivity() {
        if (cardinality == -1) {
            return;
        }
        applySelectivity();
    }

    // 根据'preConjunctCardinality'计算应用连接子集后的基数，带有'selectivity'。
    private void applySelectivity() {
        double selectivity = computeSelectivity();
        Preconditions.checkState(cardinality >= 0);
        double preConjunctCardinality = cardinality;
        cardinality = Math.round(cardinality * selectivity);
        // 不要将基数四舍五入为零以确保安全。
        if (cardinality == 0 && preConjunctCardinality > 0) {
            cardinality = 1;
        }
    }

    /**
     * 基于planNodeId递归地查找planNode
     */
    public static PlanNode findPlanNodeFromPlanNodeId(PlanNode root, PlanNodeId id) {
        if (root == null || root.getId() == null || id == null) {
            return null;
        } else if (root.getId().equals(id)) {
            return root;
        } else {
            for (PlanNode child : root.getChildren()) {
                PlanNode retNode = findPlanNodeFromPlanNodeId(child, id);
                if (retNode != null) {
                    return retNode;
                }
            }
            return null;
        }
    }

    public String getPlanTreeExplainStr() {
        StringBuilder sb = new StringBuilder();
        sb.append("[").append(getId().asInt()).append(": ").append(getPlanNodeName()).append("]");
        sb.append("\n[Fragment: ").append(getFragmentSeqenceNum()).append("]");
        sb.append("\n").append(getNodeExplainString("", TExplainLevel.BRIEF));
        return sb.toString();
    }

    public ScanNode getScanNodeInOneFragmentBySlotRef(SlotRef slotRef) {
        TupleId tupleId = slotRef.getDesc().getParent().getId();
        if (this instanceof ScanNode && tupleIds.contains(tupleId)) {
            return (ScanNode) this;
        } else if (this instanceof HashJoinNode) {
            HashJoinNode hashJoinNode = (HashJoinNode) this;
            SlotRef inputSlotRef = hashJoinNode.getMappedInputSlotRef(slotRef);
            if (inputSlotRef != null) {
                for (PlanNode planNode : children) {
                    ScanNode scanNode = planNode.getScanNodeInOneFragmentBySlotRef(inputSlotRef);
                    if (scanNode != null) {
                        return scanNode;
                    }
                }
            } else {
                return null;
            }
        } else if (!(this instanceof ExchangeNode)) {
            for (PlanNode planNode : children) {
                ScanNode scanNode = planNode.getScanNodeInOneFragmentBySlotRef(slotRef);
                if (scanNode != null) {
                    return scanNode;
                }
            }
        }
        return null;
    }

    public SlotRef findSrcSlotRef(SlotRef slotRef) {
        if (slotRef.getSrcSlotRef() != null) {
            slotRef = slotRef.getSrcSlotRef();
        }
        if (slotRef.getTable() instanceof OlapTable) {
            return slotRef;
        }
        if (this instanceof HashJoinNode) {
            HashJoinNode hashJoinNode = (HashJoinNode) this;
            SlotRef inputSlotRef = hashJoinNode.getMappedInputSlotRef(slotRef);
            if (inputSlotRef != null) {
                return hashJoinNode.getChild(0).findSrcSlotRef(inputSlotRef);
            } else {
                return slotRef;
            }
        }
        return slotRef;
    }

    protected void addRuntimeFilter(RuntimeFilter filter) {
        runtimeFilters.add(filter);
    }

    protected Collection<RuntimeFilter> getRuntimeFilters() {
        return runtimeFilters;
    }

    public void clearRuntimeFilters() {
        runtimeFilters.clear();
    }

    protected String getRuntimeFilterExplainString(boolean isBuildNode, boolean isBrief) {
        if (runtimeFilters.isEmpty()) {
            return "";
        }
        List<String> filtersStr = new ArrayList<>();
        for (RuntimeFilter filter : runtimeFilters) {
            filtersStr.add(filter.getExplainString(isBuildNode, isBrief, getId()));
        }
        return Joiner.on(", ").join(filtersStr) + "\n";
    }

    protected String getRuntimeFilterExplainString(boolean isBuildNode) {
        return getRuntimeFilterExplainString(isBuildNode, false);
    }

    /**
     * 如果计划节点实现此方法，则计划节点本身支持投影优化。
     * @param requiredSlotIdSet: 当前计划节点的上层计划节点的要求槽集合。
     *                        当上层计划节点无法计算要求槽时，requiredSlotIdSet可能为空。
     * @param analyzer 分析器
     * @throws NotImplementedException 未实现异常
     *
     * 例如：
     * 查询：select a.k1 from a, b where a.k1=b.k1
     * 计划节点树：
     *     输出表达式：a.k1
     *           |
     *     哈希连接节点
     *   （输入槽：a.k1, b.k1）
     *        |      |
     *  扫描a(k1)   扫描b(k1)
     *
     * 函数参数：requiredSlotIdSet = a.k1
     * 函数之后：
     *     哈希连接节点
     *   （输出槽：a.k1）
     *   （输入槽：a.k1, b.k1）
     */
    public void initOutputSlotIds(Set<SlotId> requiredSlotIdSet, Analyzer analyzer) throws NotImplementedException {
        throw new NotImplementedException("The `initOutputSlotIds` hasn't been implemented in " + planNodeName);
    }

    public void projectOutputTuple() throws NotImplementedException {
        throw new NotImplementedException("The `projectOutputTuple` hasn't been implemented in " + planNodeName + ". "
            + "But it does not affect the project optimizer");
    }

    /**
     * 如果计划节点实现此方法，则其子计划节点能够实现投影。
     * 此方法的返回值将用作子计划节点方法initOutputSlotIds的输入(requiredSlotIdSet)。
     * 即，仅当计划节点实现此方法时，其子节点才能实现投影优化。
     *
     * @return 计划节点的requiredSlotIdSet
     * @throws NotImplementedException 未实现异常
     * 计划节点树：
     *         聚合节点（按a.k1分组）
     *           |
     *     哈希连接节点（a.k1=b.k1）
     *        |      |
     *  扫描a(k1)   扫描b(k1)
     * 函数之后：
     *         聚合节点
     *    （所需槽：a.k1）
     */
    public Set<SlotId> computeInputSlotIds(Analyzer analyzer) throws NotImplementedException {
        throw new NotImplementedException("The `computeInputSlotIds` hasn't been implemented in " + planNodeName);
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("[").append(getId().asInt()).append(": ").append(getPlanNodeName()).append("]");
        sb.append("\nFragment: ").append(getFragmentId().asInt()).append("]");
        sb.append("\n").append(getNodeExplainString("", TExplainLevel.BRIEF));
        return sb.toString();
    }

    /**
     * 为新优化器生成的节点补充所需信息。
     */
    public void finalizeForNereids() throws UserException {

    }

    public void setOutputTupleDesc(TupleDescriptor outputTupleDesc) {
        this.outputTupleDesc = outputTupleDesc;
    }

    public TupleDescriptor getOutputTupleDesc() {
        return outputTupleDesc;
    }

    public void setProjectList(List<Expr> projectList) {
        this.projectList = projectList;
    }

    public List<Expr> getProjectList() {
        return projectList;
    }

    public List<SlotId> getOutputSlotIds() {
        return outputSlotIds;
    }

    public void setConjuncts(Set<Expr> exprs) {
        conjuncts = new ArrayList<>(exprs);
    }

    public void setCardinalityAfterFilter(long cardinalityAfterFilter) {
        this.cardinalityAfterFilter = cardinalityAfterFilter;
    }

    protected TPushAggOp pushDownAggNoGroupingOp = TPushAggOp.NONE;

    public void setPushDownAggNoGrouping(TPushAggOp pushDownAggNoGroupingOp) {
        this.pushDownAggNoGroupingOp = pushDownAggNoGroupingOp;
    }

    public void setChildrenDistributeExprLists(List<List<Expr>> childrenDistributeExprLists) {
        this.childrenDistributeExprLists = childrenDistributeExprLists;
    }

    public TPushAggOp getPushDownAggNoGroupingOp() {
        return pushDownAggNoGroupingOp;
    }

    public boolean pushDownAggNoGrouping(FunctionCallExpr aggExpr) {
        return false;
    }

    public boolean pushDownAggNoGroupingCheckCol(FunctionCallExpr aggExpr, Column col) {
        return false;
    }

    public void setNereidsId(int nereidsId) {
        this.nereidsId = nereidsId;
    }

    public void addIntermediateOutputTupleDescList(TupleDescriptor tupleDescriptor) {
        intermediateOutputTupleDescList.add(tupleDescriptor);
    }

    public void addIntermediateProjectList(List<Expr> exprs) {
        intermediateProjectListList.add(exprs);
    }
}
