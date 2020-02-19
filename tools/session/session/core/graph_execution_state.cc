#include "session/core/graph_execution_state.h"
namespace{
}

/* static */ Status GraphExecutionState::MakeForBaseGraph(
        GraphDef&& graph_def, const GraphExecutionStateOptions& options,
        std::unique_ptr<GraphExecutionState>* out_state) {
    // #ifndef __ANDROID__
    // VLOG(INFO) << "Graph proto is \n" << graph_def.DebugString();
    // #endif  // __ANDROID__

    // // auto flib_def = absl::make_unique<FunctionLibraryDefinition>(
    // // OpRegistry::Global(), graph_def.library());

    // // TF_RETURN_IF_ERROR(AddDefaultAttrsToGraphDef(&graph_def, *flib_def, 0));

    // if (options.session_options->config.graph_options().place_pruned_graph() ||
    // !options.session_options->config.experimental()
    // .optimize_for_static_graph()) {
    // auto ret = std::unique_ptr<GraphDef>(new GraphExecutionState(
    // std::make_unique<GraphDef>(std::move(graph_def)), std::move(flib_def),
    // options));

    // // When place_pruned_graph is true, a different Graph* will be initialized
    // // each time we prune the original graph, so there is no need to
    // // construct a Graph* in this case.
    // if (!options.session_options->config.graph_options().place_pruned_graph()) {
    // auto base_graph = absl::make_unique<Graph>(OpRegistry::Global());
    // RETURN_IF_ERROR(ConvertGraphDefToGraph({}, *ret->original_graph_def_,
    // base_graph.get()));
    // RETURN_IF_ERROR(ret->InitBaseGraph(std::move(base_graph)));
    // }
    // *out_state = std::move(ret);
    // } else {
    // auto ret = std::unique_ptr<GraphExecutionState>(
    // new GraphExecutionState(nullptr, std::move(flib_def), options));
    // auto base_graph = std::make_unique<Graph>(OpRegistry::Global());
    // RETURN_IF_ERROR(
    // ConvertGraphDefToGraph({}, std::move(graph_def), base_graph.get()));
    // RETURN_IF_ERROR(ret->InitBaseGraph(std::move(base_graph)));
    // *out_state = std::move(ret);
    // }
    return Status::OK();
}

Status GraphExecutionState::InitBaseGraph(std::unique_ptr<Graph>&& new_graph) {
    // Save stateful placements before placing.
    // RestoreStatefulNodes(new_graph.get());

    // GraphOptimizationPassOptions optimization_options;
    // optimization_options.session_handle = session_handle_;
    // optimization_options.session_options = session_options_;
    // optimization_options.graph = &new_graph;
    // optimization_options.flib_def = flib_def_.get();
    // optimization_options.device_set = device_set_;

    // TF_RETURN_IF_ERROR(OptimizationPassRegistry::Global()->RunGrouping(
    // OptimizationPassRegistry::PRE_PLACEMENT, optimization_options));

    // Placer placer(new_graph.get(), "", flib_def_.get(), device_set_,
    // [> default_local_device= <] nullptr,
    // session_options_ == nullptr ||
    // session_options_->config.allow_soft_placement(),
    // session_options_ != nullptr &&
    // session_options_->config.log_device_placement());
    // // TODO(mrry): Consider making the Placer cancelable.
    // TF_RETURN_IF_ERROR(placer.Run());

    // TF_RETURN_IF_ERROR(OptimizationPassRegistry::Global()->RunGrouping(
    // OptimizationPassRegistry::POST_PLACEMENT, optimization_options));

    // for (const Node* n : new_graph->nodes()) {
    // VLOG(2) << "Mapping " << n->name() << " to " << n->cost_id();
    // node_name_to_cost_id_map_[n->name()] = n->cost_id();
    // }

    // SaveStatefulNodes(new_graph.get());
    // graph_ = new_graph.release();
    return Status::OK();
}

Status GraphExecutionState::Extend(
        const GraphDef& extension_def,
        std::unique_ptr<GraphExecutionState>* out) const {
    // if (session_options_->config.experimental().optimize_for_static_graph()) {
    // return errors::FailedPrecondition(
    // "Extending the graph is not supported when "
    // "`optimize_for_static_graph` is true.");
    // }

    // GraphDef gdef;

    // // 1. Copy the function library.
    // TF_RETURN_IF_ERROR(flib_def_->AddLibrary(extension_def.library()));
    // *gdef.mutable_library() = flib_def_->ToProto();

    // // 2. Build an index of the new node names.
    // std::unordered_set<string> new_names;
    // for (const NodeDef& node : extension_def.node()) {
    // new_names.insert(node.name());
    // }

    // // 3. Add the non-duplicates from the old graph to the new graph.
    // //    Return an error if the same node name appears in both the
    // //    old graph and the extension.
    // for (const NodeDef& node : original_graph_def_->node()) {
    // if (new_names.count(node.name()) == 0) {
    // *gdef.add_node() = node;
    // } else {
    // return errors::InvalidArgument(
    // "GraphDef argument to Extend includes node '", node.name(),
    // "', which was created by a previous call to Create or Extend in this "
    // "session.");
    // }
    // }

    // // 4. Merge the versions field.
    // int old_node_size = gdef.node_size();
    // gdef.mutable_node()->MergeFrom(extension_def.node());
    // TF_RETURN_IF_ERROR(
    // AddDefaultAttrsToGraphDef(&gdef, *flib_def_, old_node_size));
    // // Merge versions
    // if (gdef.has_versions()) {
    // if (gdef.versions().producer() != extension_def.versions().producer()) {
    // return errors::InvalidArgument(
    // "Can't extend GraphDef at version ", gdef.versions().producer(),
    // " with graph at version ", extension_def.versions().producer());
    // }
    // VersionDef* versions = gdef.mutable_versions();
    // versions->set_min_consumer(std::max(
    // versions->min_consumer(), extension_def.versions().min_consumer()));
    // if (extension_def.versions().bad_consumers_size()) {
    // // Add new bad_consumers that aren't already marked bad.
    // //
    // // Note: This implementation is quadratic time if there are many calls to
    // // ExtendLocked with many bad consumers.  Since this is unlikely, and
    // // fixing it would require data structures outside of this routine,
    // // quadratic time it is.
    // auto* bad_consumers = versions->mutable_bad_consumers();
    // const std::unordered_set<int> existing(bad_consumers->begin(),
    // bad_consumers->end());
    // for (const int v : extension_def.versions().bad_consumers()) {
    // if (existing.find(v) == existing.end()) {
    // bad_consumers->Add(v);
    // }
    // }
    // }

    // } else {
    // gdef.mutable_versions()->CopyFrom(extension_def.versions());
    // }

    // // 5. Validate that the final graphdef is valid.
    // if (gdef.versions().producer() >= 5) {
    // // Validate the graph: we assume that merging two valid graphs
    // // should maintain graph validity.
    // TF_RETURN_IF_ERROR(graph::ValidateGraphDef(gdef, *flib_def_));
    // }

    // // 6. Add the extension.
    // GraphExecutionStateOptions combined_options;
    // combined_options.device_set = device_set_;
    // combined_options.session_options = session_options_;
    // combined_options.session_handle = session_handle_;
    // combined_options.stateful_placements = stateful_placements_;

    // TF_RETURN_IF_ERROR(AddDefaultAttrsToGraphDef(&gdef, *flib_def_, 0));
    // auto flib_def = absl::make_unique<FunctionLibraryDefinition>(
    // OpRegistry::Global(), gdef.library());
    // auto new_execution_state = absl::WrapUnique(
    // new GraphExecutionState(absl::make_unique<GraphDef>(std::move(gdef)),
    // std::move(flib_def), combined_options));

    // if (!session_options_->config.graph_options().place_pruned_graph()) {
    // auto base_graph = absl::make_unique<Graph>(OpRegistry::Global());
    // TF_RETURN_IF_ERROR(ConvertGraphDefToGraph(
    // {}, *new_execution_state->original_graph_def_, base_graph.get()));
    // TF_RETURN_IF_ERROR(
    // new_execution_state->InitBaseGraph(std::move(base_graph)));
    // }
    // *out = std::move(new_execution_state);

    // NOTE(mrry): Extend() is likely to be used for non-throughput-sensitive
    // interactive workloads, but in future we may want to transfer other
    // parts of the placement and/or cost model.
    return Status::OK();
}

Status GraphExecutionState::BuildGraph(const BuildGraphOptions& options,
        std::unique_ptr<ClientGraph>* out){
    LOG(INFO)<<"BuildGraph";
    const uint64_t start_time_usecs = Env::Default()->NowMicros();
    if (!graph_) {
        // It is only valid to call this method directly when the original graph
        // was created with the option `place_pruned_graph == false`.
        return errors::Internal(
                "Attempted to prune a graph that has not been fully initialized.");
    }
    // Grappler optimization might change the structure of a graph itself, and
    // also it can add/prune functions to/from the library.
    std::unique_ptr<Graph> optimized_graph;
    std::unique_ptr<FunctionLibraryDefinition> optimized_flib;

    Status s = OptimizeGraph(options, &optimized_graph, &optimized_flib);

    if (!s.ok()) {
        LOG(INFO) << "Grappler optimization failed. Error: " << s.error_message();
        // Simply copy the original graph and the function library if we couldn't
        // optimize it.
        optimized_graph.reset(new Graph(flib_def_.get()));
        CopyGraph(*graph_, optimized_graph.get());
        optimized_flib.reset(new FunctionLibraryDefinition(*flib_def_));
    }
    subgraph::RewriteGraphMetadata rewrite_metadata;
    if (session_options_ == nullptr ||
            !session_options_->config.graph_options().place_pruned_graph()) {
        RETURN_IF_ERROR(PruneGraph(options, optimized_graph.get(), &rewrite_metadata));
    } else {
        // This GraphExecutionState represents a graph that was
        // pruned when this was constructed, so we copy the metadata from
        // a member variable.
        CHECK(rewrite_metadata_);
        rewrite_metadata = *rewrite_metadata_;
    }

    int64_t collective_graph_key = BuildGraphOptions::kNoCollectiveGraphKey;

    auto end_time_usecs = Env::Default()->NowMicros() - start_time_usecs;
    std::unique_ptr<ClientGraph> dense_copy(new ClientGraph(std::move(optimized_flib),
                rewrite_metadata.feed_types, rewrite_metadata.fetch_types, collective_graph_key));
    CopyGraph(*optimized_graph, &dense_copy->graph);
    *out = std::move(dense_copy);
    return Status::OK();
}

Status GraphExecutionState::OptimizeGraph(
        const BuildGraphOptions& options, std::unique_ptr<Graph>* optimized_graph,
        std::unique_ptr<FunctionLibraryDefinition>* optimized_flib){
    if (session_options_->config.graph_options().place_pruned_graph()) {
        return errors::InvalidArgument("Can't optimize a pruned graph");
    }
    optimized_flib->reset(new FunctionLibraryDefinition(*flib_def_));
    optimized_graph->reset(new Graph(OpRegistry::Global()));
    return Status::OK();
}

Status GraphExecutionState::PruneGraph(const BuildGraphOptions& options, Graph* graph,
                subgraph::RewriteGraphMetadata* out_rewrite_metadata){
    if(options.use_function_convention){
    }else{
    }

    return Status::OK();
}
