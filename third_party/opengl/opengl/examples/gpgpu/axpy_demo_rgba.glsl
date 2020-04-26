// shader for RGBA data
// and texture2D

uniform sampler2D textureY;
uniform sampler2D textureX;
uniform float alpha;

void main(void) {
    vec4 y = texture2D(
            textureY,
            gl_TexCoord[0].st);
    vec4 x = texture2D(
            textureX
            gl_TexCoord[0].st);
    gl_FragColor =
        y + alpha*x;
}
