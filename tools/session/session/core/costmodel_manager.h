#ifndef SESSION_CORE_COSTMODEL_MANAGER_H_
#define SESSION_CORE_COSTMODEL_MANAGER_H_
#include <unordered_map>

#include "session/core/graph.h"
#include "session/utils/status.h"
#include "session/core/costmodel.h"

#include "cost_graph.pb.h"

// Used to manage all the cost models for a session.
class CostModelManager {
 public:
  ~CostModelManager();

  typedef std::unordered_map<const Graph*, CostModel*> CostModelMap;
  typedef CostModelMap::iterator CostModelMapIter;

  void ExportCostModels(CostModelMap* cost_models) {
    *cost_models = cost_models_;
  }

  CostModel* FindOrCreateCostModel(const Graph* graph);

  bool RemoveCostModelForGraph(const Graph* graph);

  Status AddToCostGraphDef(const Graph* graph, CostGraphDef* cost_graph);

 private:
  CostModelMap cost_models_ ;
};

#endif
