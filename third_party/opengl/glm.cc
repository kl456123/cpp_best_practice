#include <iostream>

using namespace std;

#include <glm/glm.hpp>
#include <glm/vec3.hpp>
#include <glm/geometric.hpp>

int main(){
    glm::vec4 Position = glm::vec4(glm::vec3(0.0), 1.0);
    glm::mat4 Model = glm::mat4(1.0);
    Model[4] = glm::vec4(1.0, 1.0, 0.0, 1.0);
    glm::vec4 Transformed = Model * Position;
    cout<<Transformed[3]<<endl;
    // std::cout<<Transformed<<std::endl;
    return 0;
}
