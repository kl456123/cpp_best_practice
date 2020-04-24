/**************************************
 * This examples is to demostrate the use of egl to run in embedding system
 * It's mainly different from glfw during initialization time.
 */
#include <iostream>
#include <string>
#include <sstream>

// #include <GL/glew.h>

#include <EGL/egl.h>
#include <GLES2/gl2.h>
#include <glog/logging.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>



void egl_init(){
    // init for embedding platform
    // just assign for the following variable
    EGLContext egl_context;
    EGLDisplay egl_display;
    EGLSurface egl_surface;

    if(eglGetCurrentContext()==EGL_NO_CONTEXT){
        egl_context = EGL_NO_CONTEXT;
        VLOG(1)<<"No Current Context Found! Need to Create Again";
    }

    egl_display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (egl_display == EGL_NO_DISPLAY) {
        LOG(FATAL)<<"eglGetDisplay Failed When Creating Context!";
    }
    int majorVersion;
    int minorVersion;
    eglInitialize(egl_display, &majorVersion, &minorVersion);
    EGLint numConfigs;
    static const EGLint configAttribs[] = {EGL_SURFACE_TYPE,
        EGL_PBUFFER_BIT,
        EGL_RENDERABLE_TYPE,
        EGL_OPENGL_ES2_BIT,
        EGL_RED_SIZE,
        8,
        EGL_GREEN_SIZE,
        8,
        EGL_BLUE_SIZE,
        8,
        EGL_ALPHA_SIZE,
        8,
        EGL_NONE};

    EGLConfig surfaceConfig;
    if(!eglChooseConfig(egl_display, configAttribs, &surfaceConfig, 1, &numConfigs)){
        eglMakeCurrent(egl_display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
        eglTerminate(egl_display);
        egl_display = EGL_NO_DISPLAY;
        LOG(FATAL)<<"eglChooseConfig Failed When Creating Context!";
    }

    static const EGLint contextAttribs[] = {EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE};
    egl_context                             = eglCreateContext(egl_display, surfaceConfig, NULL, contextAttribs);
    static const EGLint surfaceAttribs[] = {EGL_WIDTH, 1, EGL_HEIGHT, 1, EGL_NONE};
    egl_surface                             = eglCreatePbufferSurface(egl_display, surfaceConfig, surfaceAttribs);
    eglMakeCurrent(egl_display, egl_surface, egl_surface, egl_context);
    eglBindAPI(EGL_OPENGL_ES_API);
    int major;
    // glGetIntegerv(GL_MAJOR_VERSION, &major);
    // LOG(INFO)<<"current opengl version: "<<major;
}

const GLchar* vertexShaderSource[] = {
    "uniform mat4 uMVPMatrix;\n"
        "attribute vec4 aPosition;\n"
        "attribute vec2 aTexCoord;\n"
        "varying vec2 vTexCoord;\n"
        "void main() {\n"
        "gl_Position = uMVPMatrix * aPosition;\n"
        "vTexCoord = aTexCoord;\n"
        "}\n"
};

const GLchar* fragmentShaderSource[] = {
    "#extension GL_OES_EGL_image_external : require\n"
        "precision mediump float;\n"
        "uniform sampler2D sTexture;\n"
        "varying vec2 vTexCoord;\n"
        "void main() {\n"
        "       gl_FragColor = vec4(abs(vTexCoord.s), abs(vTexCoord.t), 1.0, 1.0);\n"
        "}\n"
};

float cubeVertices[] = {
    -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, -1, 1, 1, 1, -1, 1, -1, -1,
    1, -1, -1, -1, -1, -1, 1, 1, -1, -1, 1, -1,
    -1, 1, -1, -1, -1, -1, -1, 1, 1, -1, -1, 1,
    -1, -1, 1, -1, -1, -1, 1, -1, 1, 1, -1, -1,
    1, -1, -1, -1, 1, 1,
    -1, 1, 1, 1, 1, 1, -1, 1, -1, 1, 1, -1
};

float cubeTexCoords[] = {
    0, 0, 1, 0, 0, 1, 1, 1,
    0, 1, 0, 0, 1, 1, 1, 0,
    0, 0, 1, 0, 0, 1, 1, 1,
    0, 1, 0, 0, 1, 1, 1, 0,
    0, 1, 0, 0, 1, 1, 1, 0,
    1, 0, 0, 0,
    0, 0, 1, 0, 0, 1, 1, 1
};


struct ShaderHandles {
    GLuint aPosition;
    GLuint aTexCoord;
    GLuint mvMatrix;
    GLuint mvpMatrix;
};

struct Matrices {
    glm::mat4x4 modelView;
    glm::mat4x4 modelViewProjection;
};

namespace {
    glm::vec3 rotation;

    GLuint shader;
    ShaderHandles shaderHandles;
    Matrices matrices;

    bool quit = false;

    int winWidth = 640;
    int winHeight = 360;

    long long qpcFrequency;
    double timeFactor = 1.0f;

    inline float deg_to_rad(float rad) {
        return rad * float(M_PI / 180);
    }
}


GLuint compileShaders(const GLchar** vertexShaderSource, const GLchar** fragmentShaderSource)
{
    //Compile vertex shader
    GLuint vertexShader(glCreateShader(GL_VERTEX_SHADER));
    glShaderSource(vertexShader, 1, vertexShaderSource, NULL);
    glCompileShader(vertexShader);
    int logLength;
    glGetShaderiv(vertexShader, GL_INFO_LOG_LENGTH, &logLength);
    if (logLength > 0) {
        GLchar* strInfoLog = new GLchar[logLength + 1];
        glGetShaderInfoLog(vertexShader, logLength, NULL, strInfoLog);
        std::cout << "vertex shader error log:\n" << strInfoLog << "\n";
    };

    //Compile fragment shader
    GLuint fragmentShader(glCreateShader(GL_FRAGMENT_SHADER));
    glShaderSource(fragmentShader, 1, fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);
    if (logLength > 0) {
        GLchar* strInfoLog = new GLchar[logLength + 1];
        glGetShaderInfoLog(fragmentShader, logLength, NULL, strInfoLog);
        std::cout << "fragment shader error log:\n" << strInfoLog << "\n";
    };

    //Link vertex and fragment shader together
    GLuint program(glCreateProgram());
    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);
    glLinkProgram(program);

    //Delete shaders objects
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    return program;
}

void glErrorLog() {
    GLuint glError = glGetError();
    if (glError != GL_NO_ERROR) {
        std::cout << "OpenGL Error : " << glError << "\n";
    }
}

void initScene() {
    // QueryPerformanceCounter((LARGE_INTEGER*)&qpcFrequency);

    glEnable(GL_DEPTH_TEST);
    glClearColor(0.0f, 0.0f, 0.2f, 1.0f);

    rotation = glm::vec3(15.0, 45.0, 0.0);

    shader = compileShaders(vertexShaderSource, fragmentShaderSource);
    glUseProgram(shader);

    shaderHandles.aPosition = glGetAttribLocation(shader, "aPosition");
    shaderHandles.aTexCoord = glGetAttribLocation(shader, "aTexCoord");

    shaderHandles.mvpMatrix = glGetUniformLocation(shader, "uMVPMatrix");
    shaderHandles.mvMatrix = glGetUniformLocation(shader, "uMVMatrix");

    glVertexAttribPointer(shaderHandles.aPosition, 3, GL_FLOAT, false, 0, &cubeVertices);
    glVertexAttribPointer(shaderHandles.aTexCoord, 2, GL_FLOAT, false, 0, &cubeTexCoords);
    glEnableVertexAttribArray(shaderHandles.aPosition);
    glEnableVertexAttribArray(shaderHandles.aTexCoord);
}

void renderScene(double timeFactor) {
    glViewport(0, 0, winWidth, winHeight);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    matrices.modelView = glm::mat4();
    matrices.modelView = glm::translate(matrices.modelView, glm::vec3(0.0f, 0.0f, -6.0f));
    matrices.modelView = glm::rotate(matrices.modelView, deg_to_rad(rotation.x), glm::vec3(1.0f, 0.0f, 0.0f));
    matrices.modelView = glm::rotate(matrices.modelView, deg_to_rad(rotation.y), glm::vec3(0.0f, 1.0f, 0.0f));

    matrices.modelViewProjection = glm::perspective(45.0f, (float)winWidth / (float)winHeight, 0.1f, 100.0f) * matrices.modelView;

    glUniformMatrix4fv(shaderHandles.mvpMatrix, 1, false, &matrices.modelViewProjection[0][0]);
    glUniformMatrix4fv(shaderHandles.mvMatrix, 1, false, &matrices.modelView[0][0]);

    glDrawArrays(GL_TRIANGLE_STRIP, 0, 26);

    glErrorLog();

    rotation.y += (rotation.y < 360.0) ? timeFactor : -360.0f + timeFactor;
}

void outputGLESInfo() {
    std::cout << "GL_VENDOR = " << glGetString(GL_VENDOR) << "\n";
    std::cout << "GL_RENDERER = " << glGetString(GL_RENDERER) << "\n";
    std::cout << "GL_VERSION = " << glGetString(GL_VERSION) << "\n";
    std::cout << "GL_SHADING_LANGUAGE_VERSION = " << glGetString(GL_SHADING_LANGUAGE_VERSION) << "\n";
    std::cout << "Extensions :\n";
    std::string extBuffer;
    std::stringstream extStream;
    extStream << glGetString(GL_EXTENSIONS);
    while (extStream >> extBuffer) {
        std::cout << extBuffer << "\n";
    }

}

int main(int arc, char* argv[]){
    google::InitGoogleLogging(argv[0]);
    // egl_init();
    // printf("GL_VERSION  : %s\n", glGetString(GL_VERSION));
    // printf("GLSL ES VERSION  :%s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));

    // HWND hwnd = createWindow(winWidth, winHeight);
    // HDC hdc = GetDC(hwnd);

    EGLDisplay eglDisplay = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (eglDisplay == EGL_NO_DISPLAY) {
        std::cout << "Could not get egl display!" << std::endl;
        return 1;
    }

    EGLint eglVersionMajor, eglVersionMinor;
    eglInitialize(eglDisplay, &eglVersionMajor, &eglVersionMinor);
    eglBindAPI(EGL_OPENGL_ES_API);

    EGLint configAttributes[] =
    {
        EGL_BUFFER_SIZE, 0,
        EGL_RED_SIZE, 5,
        EGL_GREEN_SIZE, 6,
        EGL_BLUE_SIZE, 5,
        EGL_ALPHA_SIZE, 0,
        EGL_COLOR_BUFFER_TYPE, EGL_RGB_BUFFER,
        EGL_CONFIG_CAVEAT, EGL_DONT_CARE,
        EGL_CONFIG_ID, EGL_DONT_CARE,
        EGL_DEPTH_SIZE, 24,
        EGL_LEVEL, 0,
        EGL_MAX_SWAP_INTERVAL, EGL_DONT_CARE,
        EGL_MIN_SWAP_INTERVAL, EGL_DONT_CARE,
        EGL_NATIVE_RENDERABLE, EGL_DONT_CARE,
        EGL_NATIVE_VISUAL_TYPE, EGL_DONT_CARE,
        EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT,
        EGL_SAMPLE_BUFFERS, 0,
        EGL_SAMPLES, 0,
        EGL_STENCIL_SIZE, 0,
        EGL_SURFACE_TYPE, EGL_WINDOW_BIT,
        EGL_TRANSPARENT_TYPE, EGL_NONE,
        EGL_TRANSPARENT_RED_VALUE, EGL_DONT_CARE,
        EGL_TRANSPARENT_GREEN_VALUE, EGL_DONT_CARE,
        EGL_TRANSPARENT_BLUE_VALUE, EGL_DONT_CARE,
        EGL_NONE
    };

    EGLint surfaceAttributes[] = { EGL_NONE };
    EGLint contextAttributes[] = { EGL_CONTEXT_CLIENT_VERSION, 2, EGL_NONE };

    EGLint nrOfConfigs;
    EGLConfig windowConfig;
    eglChooseConfig(eglDisplay, configAttributes, &windowConfig, 1, &nrOfConfigs);

    // EGLSurface eglSurface = eglCreateWindowSurface(eglDisplay, windowConfig, hwnd, surfaceAttributes);
    EGLSurface eglSurface = eglCreatePbufferSurface(eglDisplay, windowConfig, surfaceAttributes);
    if (eglSurface == EGL_NO_SURFACE) {
        std::cerr << "Could not create EGL surface : " << eglGetError() << std::endl;
        return 1;
    }

    EGLContext eglContext = eglCreateContext(eglDisplay, windowConfig, NULL, contextAttributes);

    if (eglContext == EGL_NO_CONTEXT) {
        std::cout << "Could not create egl context : " << eglGetError() << std::endl;
        return 1;
    }

    std::cout << "EGL Version = " << eglQueryString(eglDisplay, EGL_VERSION) << "\n";
    std::cout << "EGL Vendor = " << eglQueryString(eglDisplay, EGL_VENDOR) << "\n";
    std::cout << "EGL Client APIs : \n" << eglQueryString(eglDisplay, EGL_CLIENT_APIS) << "\n";
    std::cout << "EGL Extensions : \n" << eglQueryString(eglDisplay, EGL_EXTENSIONS) << "\n";

    std::cout << "EGL Configurations:\n";

    EGLConfig *eglConfigs;
    int eglNumConfigs;
    eglGetConfigs(eglDisplay, NULL, 0, &eglNumConfigs);
    eglConfigs = new EGLConfig[eglNumConfigs];

    for (int i = 0; i < eglNumConfigs; i++) {
        std::cout << "Config " << i << "\n";
        std::cout << "Supported APIs :\n";
        int eglRenderable;
        eglGetConfigAttrib(eglDisplay, eglConfigs[i], EGL_RENDERABLE_TYPE, &eglRenderable);
        if (eglRenderable & EGL_OPENGL_ES_BIT) std::cout << "OPENGL ES" << "\n";
        if (eglRenderable & EGL_OPENGL_ES2_BIT) std::cout << "OPENGL ES2" << "\n";
        if (eglRenderable & EGL_OPENVG_BIT) std::cout << "OPENVG" << "\n";
        if (eglRenderable & EGL_OPENGL_BIT) std::cout << "OPENGL" << "\n";
        std::cout << "\n";
    }

    EGLint attr[] = {
        EGL_BUFFER_SIZE, 16,
        EGL_RENDERABLE_TYPE,
        EGL_OPENGL_ES2_BIT,
        EGL_NONE
    };

    EGLConfig *eglConfig = new EGLConfig;
    int elgNumConfig;
    if (!eglChooseConfig(eglDisplay, attr, eglConfig, sizeof(eglConfig), &eglNumConfigs)) {
        std::cout << "Could not get valid egl configuration!" << std::endl;
        return 1;
    }

    eglMakeCurrent(eglDisplay, eglSurface, eglSurface, eglContext);
    outputGLESInfo();
    initScene();

    // Render loop

    // MSG uMsg;
    // PeekMessage(&uMsg, NULL, 0, 0, PM_REMOVE);

    long long qpcStart, qpcEnd;
    while (!quit)  {

        // QueryPerformanceCounter((LARGE_INTEGER*)&qpcStart);
        renderScene(timeFactor);

        // while (PeekMessage(&uMsg, NULL, 0, 0, PM_REMOVE) > 0) {
            // TranslateMessage(&uMsg);
            // DispatchMessage(&uMsg);
        // }

        eglSwapBuffers(eglDisplay, eglSurface);
        // QueryPerformanceCounter((LARGE_INTEGER*)&qpcEnd);
        double dTime = (double)(qpcEnd - qpcStart) / (double)qpcFrequency;
        timeFactor += dTime * 0.01f;


    }
    return 0;
}
