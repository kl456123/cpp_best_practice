#include <sstream>

#include "opengl/core/opengl.h"
#include "opengl/utils/util.h"
#include "opengl/utils/logging.h"


namespace opengl{
    void setLocalSize(std::vector<std::string>& prefix, int* localSize,
            std::vector<int> local_sizes){
        GLint maxLocalSizeX, maxLocalSizeY, maxLocalSizeZ;
        glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 0, &maxLocalSizeX);
        glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 1, &maxLocalSizeY);
        glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 2, &maxLocalSizeZ);

        localSize[0]     = local_sizes[0] < maxLocalSizeX ? local_sizes[0] : maxLocalSizeX;
        localSize[1]     = local_sizes[1] < maxLocalSizeY ? local_sizes[1] : maxLocalSizeY;
        localSize[2]     = local_sizes[2] < maxLocalSizeZ ? local_sizes[2] : maxLocalSizeZ;

        {
            std::ostringstream os;
            os << "#define XLOCAL " << localSize[0];
            prefix.push_back(os.str());
        }
        {
            std::ostringstream os;
            os << "#define YLOCAL " << localSize[1];
            prefix.push_back(os.str());
        }
        {
            std::ostringstream os;
            os << "#define ZLOCAL " << localSize[2];
            prefix.push_back(os.str());
        }

    }

    IntList AmendShape(const IntList& shape, const int amend_size){
        // CHECK_LE(shape.size(), amend_size);
        IntList amended_shape;
        if(amend_size<shape.size()){
            for(int i=shape.size()-amend_size;i<shape.size();++i){
                amended_shape.emplace_back(shape[i]);
            }
            return amended_shape;
        }
        const int remain_dims = amend_size-shape.size();
        amended_shape = shape;
        for(int i=0;i<remain_dims;++i){
            amended_shape.insert(amended_shape.begin(), 1);
        }
        return amended_shape;
    }
}//namespace opengl

