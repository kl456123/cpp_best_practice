#include <sstream>

#include "opengl.h"
#include "util.h"


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
}//namespace opengl

