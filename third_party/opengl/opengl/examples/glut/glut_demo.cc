#include <GL/freeglut.h>

int glut_init(int argc, char* argv[]);


int main(int argc, char* argv[]){
    glut_init(argc, argv);
    return 0;
}

int glut_init(int argc, char* argv[]){
    glutInit(&argc, argv);

    int window = glutCreateWindow(argv[0]);
    return window;
}
