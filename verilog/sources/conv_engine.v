// conv_engine: wrapper autour de mac_unit + requantize, utilise par
// conv1/conv2/dense1/dense2. Un "terme" est un seul produit act*w par cycle
// (mac_array/LANES=32 a ete abandonne : le sommateur en arbre parallele
// cassait le timing SPI, cf. l'en-tete de mem_streamer.v).
//
//   s_valid & s_first : acc <- s_bias + terme
//   s_valid           : acc <- acc    + terme
//   s_valid & s_last  : latch (m, shift, zp_out, clamp_min), r_valid pulse
//                       4 cycles plus tard (pipeline srdhm 2 etages)

module conv_engine (
    // timing
    input wire clk,
    input wire rst,

    // data stream IN
    input wire                s_valid,     // terme valide ce cycle
    input wire signed [ 7:0]  s_act,       // activation int8 (x_q)
    input wire signed [ 7:0]  s_w,         // poids int8 (w_q)
    input wire                s_first,     // 1er terme -> acc = s_bias + terme
    input wire                s_last,      // dernier terme -> requant + emit
    input wire signed [31:0]  s_bias,      // int32 bias      (echantillonne @ s_first)
    input wire signed [31:0]  s_mult,      // Q31 multiplier  (echantillonne @ s_last)
    input wire        [ 4:0]  s_shift,     // right shift     (echantillonne @ s_last)
    input wire signed [ 7:0]  s_zp_in,     // input zero-point (chaque terme)
    input wire signed [ 7:0]  s_zp_out,    // output zero-point (echantillonne @ s_last)
    input wire signed [ 7:0]  s_clamp_min, // = zp_out couches ReLU, -128 dense2


    //result stream OUT
    output reg              r_valid,  // un pulse par element qui est termine
    output reg signed [7:0] r_data    // int8 y_q
);

    // MAC
    wire signed [31:0] acc;

    mac_unit u_mac (
        .clk       (clk),
        .rst       (rst),
        .en        (s_valid),
        .first_term(s_first),
        .act       (s_act),
        .weight    (s_w),
        .zero_point(s_zp_in),
        .bias      (s_bias),
        .acc       (acc)
    );

    //constantes de requant latched sur le dernier terme
    reg signed [31:0] mult_r;
    reg        [ 4:0] shift_r;
    reg signed [31:0] zp_out_r;  // sign-extend int8 -> int32
    reg signed [31:0] clamp_min_r;
    reg               req_pending;  // acc final present ce cycle, capture par u_req

    // req_pending suit la latence de requantize (2 cycles) jusqu'a l'emission.
    reg req_pending_d1, req_pending_d2;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            req_pending    <= 1'b0;
            req_pending_d1 <= 1'b0;
            req_pending_d2 <= 1'b0;
        end else begin
            req_pending    <= s_valid & s_last;
            req_pending_d1 <= req_pending;
            req_pending_d2 <= req_pending_d1;
            if (s_valid & s_last) begin
                mult_r      <= s_mult;
                shift_r     <= s_shift;
                zp_out_r    <= s_zp_out;
                clamp_min_r <= s_clamp_min;
            end
        end
    end

    // requantize: srdhm (2 cycles) -> round_shift -> relu_clamp. y_q valide
    // au cycle req_pending_d2.
    wire signed [31:0] y_q;

    requantize u_req (
        .clk      (clk),
        .acc      (acc),
        .m        (mult_r),
        .shift    (shift_r),
        .zp_out   (zp_out_r),
        .clamp_min(clamp_min_r),
        .y_q      (y_q)
    );

    //emission: un pulse r_valid par element, dans l'ordre
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            r_valid <= 1'b0;
            r_data  <= 8'sd0;
        end else begin
            r_valid <= req_pending_d2;
            if (req_pending_d2) r_data <= y_q[7:0];
        end
    end

endmodule
