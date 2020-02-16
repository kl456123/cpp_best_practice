#ifndef SESSION_CORE_SESSION_H_
#define SESSION_CORE_SESSION_H_
#include "graph.pb.h"
#include "device_attributes.pb.h"
#include "config.pb.h"

#include "session/core/tensor.h"
#include "session/core/session_options.h"
#include "session/utils/status.h"
#include "session/utils/errors.h"
#include "session/core/device_mgr.h"
#include "session/core/threadpool_options.h"
/// A Session allows concurrent calls to Run(), though a Session must
/// be created / extended by a single thread.
///
/// Only one thread must call Close(), and Close() must only be called
/// after all other calls to Run() have returned.
class Session {
    public:
        Session();
        virtual ~Session();

        /// \brief Create the graph to be used for the session.
        ///
        /// Returns an error if this session has already been created with a
        /// graph. To re-use the session with a different graph, the caller
        /// must Close() the session first.
        virtual Status Create(const GraphDef& graph) = 0;
        virtual Status Create(GraphDef&& graph) { return Create(graph); }

        /// \brief Adds operations to the graph that is already registered with the
        /// Session.
        ///
        /// The names of new operations in "graph" must not exist in the
        /// graph that is already registered.
        virtual Status Extend(const GraphDef& graph) = 0;
        virtual Status Extend(GraphDef&& graph) { return Extend(graph); }

        /// \brief Runs the graph with the provided input tensors and fills
        /// `outputs` for the endpoints specified in `output_tensor_names`.
        /// Runs to but does not return Tensors for the nodes in
        /// `target_node_names`.
        ///
        /// The order of tensors in `outputs` will match the order provided
        /// by `output_tensor_names`.
        ///
        /// If `Run` returns `OK()`, then `outputs->size()` will be equal to
        /// `output_tensor_names.size()`.  If `Run` does not return `OK()`, the
        /// state of `outputs` is undefined.
        ///
        /// REQUIRES: The name of each Tensor of the input or output must
        /// match a "Tensor endpoint" in the `GraphDef` passed to `Create()`.
        ///
        /// REQUIRES: At least one of `output_tensor_names` and
        /// `target_node_names` must be non-empty.
        ///
        /// REQUIRES: outputs is not nullptr if `output_tensor_names` is non-empty.
        virtual Status Run(const std::vector<std::pair<string, Tensor> >& inputs,
                const std::vector<string>& output_tensor_names,
                const std::vector<string>& target_node_names,
                std::vector<Tensor>* outputs) = 0;
        /// \brief Like `Run`, but allows users to pass in a `RunOptions` proto and
        /// to retrieve non-Tensor metadata output via a `RunMetadata` proto for this
        /// step.  `run_metadata` may be nullptr, in which case any metadata output is
        /// discarded.
        /// NOTE: This API is still experimental and may change.
        virtual Status Run(const RunOptions& run_options,
                const std::vector<std::pair<string, Tensor> >& inputs,
                const std::vector<string>& output_tensor_names,
                const std::vector<string>& target_node_names,
                std::vector<Tensor>* outputs, RunMetadata* run_metadata);

        /// \brief Like `Run` with `RunOptions` proto, but allows user to provide
        /// custom threadpool implementation via ThreadPoolOptions.
        /// NOTE: This API is still experimental and may change.
        virtual Status Run(const RunOptions& run_options,
                const std::vector<std::pair<string, Tensor> >& inputs,
                const std::vector<string>& output_tensor_names,
                const std::vector<string>& target_node_names,
                std::vector<Tensor>* outputs, RunMetadata* run_metadata,
                const thread::ThreadPoolOptions& threadpool_options) {
            return errors::Unimplemented(
                    "Run with threadpool is not supported for this session.");
        }


        ///
        /// Retrieves the list of available devices within the session, and populates
        /// *response. This API is optional. If it is unimplemented, Status will
        /// return a corresponding error message, and *response will be unmodified.
        virtual Status ListDevices(std::vector<DeviceAttributes>* response) = 0;

        /// \brief Closes this session.
        ///
        /// Closing a session releases the resources used by this session
        /// on the TensorFlow runtime (specified during session creation by
        /// the `SessionOptions::target` field).
        virtual Status Close() = 0;

        /// \brief Release global graph-related state in this session.
        ///
        /// After calling `this->Finalize()`, calls to `this->Run()` with previously
        /// unseen feeds and fetches, and calls to `this->MakeCallable()` will fail.
        /// Using `MakeCallable()` and `RunCallable()` is recommended, because
        /// explicit callable creation makes it clearer where the `Finalize()` call
        /// should be placed.
        ///
        /// This API can be used in conjunction with a "warmup" phase to reduce the
        /// memory consumed by the session:
        ///
        /// 1. Call `Session::Create()`.
        /// 2. Call `Session::MakeCallable()` for all subgraphs that you will execute
        ///    in the session.
        /// 3. Call `Session::Finalize()` to release global graph-related state.
        /// 4. Call `Session::RunCallable()` with the handle(s) created in step 2.
        ///
        /// NOTE: This API is still experimental and may change.
        virtual Status Finalize() {
            return errors::Unimplemented("Finalize is not supported for this session.");
        }

        // NOTE(ashankar): As of July 2017, this method was added to facilitate some
        // experimentation. Reconsider/re-evaluate after September 2017.
        //
        // Sets `*output` to the `DeviceMgr` that owns accessible devices in the
        // address-space of the caller.
        virtual Status LocalDeviceManager(const DeviceMgr** output) {
            return errors::Unimplemented(
                    "LocalDeviceManager is not supported for this session.");
        }
};

/// \brief Create a new session with the given options.
///
/// If session creation succeeds, the new `Session` will be stored in
/// `*out_session`, the caller will take ownership of the returned
/// `*out_session`, and this function will return `OK()`. Otherwise, this
/// function will return an error status and set *out_session to nullptr.
Status NewSession(const SessionOptions& options, Session** out_session);

Session* NewSession(const SessionOptions& options);
#endif
