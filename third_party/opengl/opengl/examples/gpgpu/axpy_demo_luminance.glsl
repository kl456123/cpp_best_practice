// shader for luminance data
// and texture rectangles

uniform samplerRect textureY;
uniform samplerRect textureX;
uniform float alpha;

void main(void) {
    float y = textureRect(
            textureY,
            gl_TexCoord[0].st).x;
    float x = textureRect(
            textureX,
            gl_TexCoord[0].st).x;
    gl_FragColor.x =
        y + alpha*x;
}
