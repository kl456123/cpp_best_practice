#include "session/core/costmodel_manager.h"


CostModelManager::~CostModelManager() {
  for (auto it : cost_models_) {
      delete it.second;
    }
}
