// (N, num_boxes, 1, num_entries)
// (N*num_boxes* 1* num_entries/4, 4) (any4)
uniform PRECISION sampler2D prediction;

// any4
// (N, H, W, 4)
uniform PRECISION sampler2D anchors;

uniform vec4 variances;
uniform ivec4 input_shape;

out vec4 color;

void main() {
    ivec2 pos = ivec2(gl_FragCoord.xy);
    // xywh
    vec4 pred_encoded_boxes = texelFetch(prediction, pos, 0);
    // xywh
    vec4 anchor_boxes = texelFetch(prediction, pos, 0);

    // decoded
    vec4 decoded_boxes_xywh;
    decoded_boxes_xywh.zw= exp(pred_encoded_boxes.zw)*variances.zw*anchor_boxes.zw;
    decoded_boxes_xywh.xy = pred_encoded_boxes.xy*variances.xy +anchor_boxes.xy;

    // xywh -> xyxy
    color.xy = decoded_boxes_xywh.xy-0.5 * decoded_boxes_xywh.zw;
    color.zw = decoded_boxes_xywh.xy+0.5 * decoded_boxes_xywh.zw;
}
