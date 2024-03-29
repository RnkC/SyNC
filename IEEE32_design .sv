`timescale 1ps/1fs

module clk_div 
#(parameter
  THRESHOLD=24// divides i_clk by 24 to obtain ck_stb which is the divided clock signal
  )
  (i_clk, ck_stb);
  
  input i_clk;
  output reg ck_stb = 0;
  
  reg [($clog2(THRESHOLD >> 1)-1):0] counter = 0;
  reg counter_reset = 0;
  
  //output reg ck_stb = 0;
  
  always @(posedge i_clk)
    counter_reset <= (counter == (THRESHOLD >> 1) - 1'b1);
  
  always @(posedge i_clk)
  begin
    if(counter_reset)
      counter <= 1;    
    else
      counter <= counter + 1;
  end
  
  always @(posedge i_clk)
    if(counter_reset)
      ck_stb <= ~ck_stb;
  
endmodule

//Clock generator module
module clk_gen
  #(
  parameter THRESHOLD_FOR_CLOCK = 24
)
  (clk_0_1ps,clk_out);
input clk_0_1ps;
output clk_out; 

 // reg clk_0_1ns;
  wire clk_out;
  clk_div #(.THRESHOLD(THRESHOLD_FOR_CLOCK)) cd (.i_clk(clk_0_1ps), .ck_stb(clk_out));
  
endmodule

module sign_bit(
output wire sign,
input wire[31:0] in1,
input wire[31:0] in2
);
xor(sign,in1[31],in2[31]);
endmodule

//1 bit Full Adder
module full_adder(
  input wire clk,
output reg sum1,
output reg cout1,
input wire in1,
input wire in2,
input wire cin
);
wire sum;
wire cout;

wire temp1;
wire temp2;
wire temp3;
xor(sum,in1,in2,cin);
and(temp1,in1,in2);
and(temp2,in1,cin);
and(temp3,in2,cin);
or(cout,temp1,temp2,temp3);

always@(posedge clk)
begin
  sum1<=sum;
  cout1<=cout;
end

endmodule
//8 bit Ripple-carry adder
module ripple_8(
input wire clk,
output reg[7:0] sum1,
output reg cout1,
input wire[7:0] in1,
input wire[7:0] in2,
input wire cin
);
wire[7:0] sum;
wire cout;
wire c1,c2,c3,c4,c5,c6,c7;
full_adder FA1(clk,sum[0],c1,in1[0],in2[0],cin);
full_adder FA2(clk,sum[1],c2,in1[1],in2[1],c1);
full_adder FA3(clk,sum[2],c3,in1[2],in2[2],c2);
full_adder FA4(clk,sum[3],c4,in1[3],in2[3],c3);
full_adder FA5(clk,sum[4],c5,in1[4],in2[4],c4);
full_adder FA6(clk,sum[5],c6,in1[5],in2[5],c5);
full_adder FA7(clk,sum[6],c7,in1[6],in2[6],c6);
full_adder FA8(clk,sum[7],cout,in1[7],in2[7],c7);
always@(posedge clk)
begin
  sum1<=sum;
  cout1<=cout;
end
endmodule

//1 bit subtractor with subtrahend = 1
module full_subtractor_sub1(
  input clk,
output reg diff1, //difference
output reg bout1, //borrow out
input wire min, //minuend
input wire bin //borrow in

);

wire diff;
wire bout;

//Here, the subtrahend is always 1. We can implement it as:
xnor(diff,min,bin);
or(bout,~min,bin);

always@(posedge clk)
begin
  diff1<=diff;
  bout1<=bout;
end

endmodule
//1 bit subtractor with subtrahend = 0
module full_subtractor_sub0(
  input clk,
output reg diff1, //difference
output reg bout1, //borrow out
input wire min, //minuend
input wire bin //borrow in
);

wire diff;
wire bout;

//Here, the subtrahend is always 0.We can implement it as:
xor(diff,min,bin);
and(bout,~min,bin);


always@(posedge clk)
begin
  diff1<=diff;
  bout1<=bout;
end

endmodule

//9 bit subtractor
module subtractor_9(
input wire clk,
output reg [8:0] diff1,
output reg bout1,
input wire [8:0] min,
input wire bin
);

wire [8:0] diff;
wire bout;

wire b1,b2,b3,b4,b5,b6,b7,b8;
full_subtractor_sub1 sub1(clk,diff[0],b1,min[0],bin);
full_subtractor_sub1 sub2(clk,diff[1],b2,min[1],b1);
full_subtractor_sub1 sub3(clk,diff[2],b3,min[2],b2);
full_subtractor_sub1 sub4(clk,diff[3],b4,min[3],b3);
full_subtractor_sub1 sub5(clk,diff[4],b5,min[4],b4);
full_subtractor_sub1 sub6(clk,diff[5],b6,min[5],b5);
full_subtractor_sub1 sub7(clk,diff[6],b7,min[6],b6);
full_subtractor_sub0 sub8(clk,diff[7],b8,min[7],b7); //Two most significand subtrahends are 0 in 001111111.
full_subtractor_sub0 sub9(clk,diff[8],bout,min[8],b8);
always@(posedge clk)
begin
  diff1<=diff;
  bout1<=bout;
end

endmodule

module block(
input wire clk, 
output reg ppo1, //output partial product term
output reg cout1, //output carry out
output reg mout1, //output multiplicand term
input wire min, //input multiplicand term
input wire ppi, //input partial product term
input wire q, //input multiplier term
input wire cin //input carry in
);
wire ppo; 
wire cout; 
wire mout;


wire temp;
and(temp,min,q);
full_adder FA(clk,ppo,cout,ppi,temp,cin);
or(mout,min,1'b0);

always@(posedge clk)
begin
  ppo1<=ppo;
  cout1<=cout;
  mout1<=mout; 
  
end
endmodule

module row(
input wire clk,  
output reg[23:0] ppo1,
output reg[23:0] mout1,
output reg sum1,
input wire[23:0] min,
input wire[23:0] ppi,
input wire q
);
wire[23:0] ppo;
wire[23:0] mout;
wire sum;

wire c1,c2,c3,c4,c5,c6,c7,c8,c9,c10;
wire c11,c12,c13,c14,c15,c16,c17,c18,c19,c20;
wire c21,c22,c23;
block b1 (clk,sum,c1,mout[0],min[0],ppi[0],q,1'b0);
block b2 (clk,ppo[0], c2, mout[1], min[1], ppi[1], q, c1);
block b3 (clk,ppo[1], c3, mout[2], min[2], ppi[2], q, c2);
block b4 (clk,ppo[2], c4, mout[3], min[3], ppi[3], q, c3);
block b5 (clk,ppo[3], c5, mout[4], min[4], ppi[4], q, c4);
block b6 (clk,ppo[4], c6, mout[5], min[5], ppi[5], q, c5);
block b7 (clk,ppo[5], c7, mout[6], min[6], ppi[6], q, c6);
block b8 (clk,ppo[6], c8, mout[7], min[7], ppi[7], q, c7);
block b9 (clk,ppo[7], c9, mout[8], min[8], ppi[8], q, c8);
block b10(clk,ppo[8], c10, mout[9], min[9], ppi[9], q, c9);
block b11(clk,ppo[9], c11, mout[10], min[10], ppi[10], q, c10);
block b12(clk,ppo[10], c12, mout[11], min[11], ppi[11], q, c11);
block b13(clk,ppo[11], c13, mout[12], min[12], ppi[12], q, c12);
block b14(clk,ppo[12], c14, mout[13], min[13], ppi[13], q, c13);
block b15(clk,ppo[13], c15, mout[14], min[14], ppi[14], q, c14);
block b16(clk,ppo[14], c16, mout[15], min[15], ppi[15], q, c15);
block b17(clk,ppo[15], c17, mout[16], min[16], ppi[16], q, c16);
block b18(clk,ppo[16], c18, mout[17], min[17], ppi[17], q, c17);
block b19(clk,ppo[17], c19, mout[18], min[18], ppi[18], q, c18);
block b20(clk,ppo[18], c20, mout[19], min[19], ppi[19], q, c19);
block b21(clk,ppo[19], c21, mout[20], min[20], ppi[20], q, c20);
block b22(clk,ppo[20], c22, mout[21], min[21], ppi[21], q, c21);
block b23(clk,ppo[21], c23, mout[22], min[22], ppi[22], q, c22);
block b24(clk,ppo[22], ppo[23], mout[23], min[23], ppi[23], q, c23);

always@(posedge clk)
begin
  ppo1<=ppo;
  mout1<=mout;
  sum1<=sum;
end
endmodule

module product(
input clk,
output reg[47:0] sum1,
input wire[23:0] min,
input wire[23:0]q
);

wire [47:0] sum;

wire [23:0] temp1,temp2,temp3,temp4,temp5,temp6,temp7,temp8,temp9,temp10; //diagonal m
wire [23:0] temp11,temp12,temp13,temp14,temp15,temp16,temp17,temp18,temp19,temp20;
wire [23:0] temp21,temp22,temp23,temp24;
wire [23:0] ptemp1,ptemp2,ptemp3,ptemp4,ptemp5,ptemp6,ptemp7,ptemp8,ptemp9,ptemp10;
//vertical p
wire [23:0] ptemp11,ptemp12,ptemp13,ptemp14,ptemp15,ptemp16,ptemp17,ptemp18,ptemp19,ptemp20;
wire [23:0] ptemp21,ptemp22,ptemp23;
row r1 (clk,ptemp1, temp1, sum[0], min, 24'h000000, q[0]);
row r2 (clk,ptemp2, temp2, sum[1], temp1, ptemp1, q[1]);
row r3 (clk,ptemp3, temp3, sum[2], temp2, ptemp2, q[2]);
row r4 (clk,ptemp4, temp4, sum[3], temp3, ptemp3, q[3]);
row r5 (clk,ptemp5, temp5, sum[4], temp4, ptemp4, q[4]);
row r6 (clk,ptemp6, temp6, sum[5], temp5, ptemp5, q[5]);
row r7 (clk,ptemp7, temp7, sum[6], temp6, ptemp6, q[6]);
row r8 (clk,ptemp8, temp8, sum[7], temp7, ptemp7, q[7]);
row r9 (clk,ptemp9, temp9, sum[8], temp8, ptemp8, q[8]);
row r10(clk,ptemp10, temp10, sum[9], temp9, ptemp9, q[9]);
row r11(clk,ptemp11, temp11, sum[10], temp10, ptemp10, q[10]);
row r12(clk,ptemp12, temp12, sum[11], temp11, ptemp11, q[11]);
row r13(clk,ptemp13, temp13, sum[12], temp12, ptemp12, q[12]);
row r14(clk,ptemp14, temp14, sum[13], temp13, ptemp13, q[13]);
row r15(clk,ptemp15, temp15, sum[14], temp14, ptemp14, q[14]);
row r16(clk,ptemp16, temp16, sum[15], temp15, ptemp15, q[15]);
row r17(clk,ptemp17, temp17, sum[16], temp16, ptemp16, q[16]);
row r18(clk,ptemp18, temp18, sum[17], temp17, ptemp17, q[17]);
row r19(clk,ptemp19, temp19, sum[18], temp18, ptemp18, q[18]);
row r20(clk,ptemp20, temp20, sum[19], temp19, ptemp19, q[19]);
row r21(clk,ptemp21, temp21, sum[20], temp20, ptemp20, q[20]);
row r22(clk,ptemp22, temp22, sum[21], temp21, ptemp21, q[21]);
row r23(clk,ptemp23, temp23, sum[22], temp22, ptemp22, q[22]);
row r24(clk,sum[47:24], temp24, sum[23], temp23, ptemp23, q[23]);

always@(posedge clk)
begin
  sum1<=sum;
end


endmodule

module normalize(
  input wire clk,
output reg[22:0] adj_mantissa1, //adjusted mantissa (after extracting out required part)
output reg norm_flag1,
input wire[47:0] prdt
); //returns norm =1 if normalization needs to be done.

wire[22:0] adj_mantissa;
wire norm_flag;

and(norm_flag,prdt[47],1'b1); //sel = 1 if leading one is at 47... needs normalization

//if sel = 0, leading zero not at 47... no need of normalization
wire [22:0] results0,results1;
assign results0 = prdt[45:23];
assign results1 = prdt[46:24]; 
assign adj_mantissa = norm_flag?results1:results0;
always@(posedge clk)
begin
  adj_mantissa1<=adj_mantissa;
  norm_flag1<=norm_flag;
end

endmodule

module controlx(
input wire clk,
input [31:0] inp1,inp2,
output wire[31:0] out,
output wire underflow,
output wire overflow
);

wire sign;
wire [7:0] exp1;
wire [7:0] exp2;
wire [7:0] exp_out;
wire [7:0] test_exp;
wire [22:0] mant1;
wire [22:0] mant2;
wire [22:0] mant_out;
sign_bit sign_bit1(sign,inp1,inp2);
wire [7:0]temp1;
wire dummy; //to connect unused cout ports of adder wire carry;
wire [8:0] sub_temp1;
reg [8:0] sub_temp;
ripple_8 rip1(clk,temp1,carry,inp1[30:23],inp2[30:23],1'b0);
subtractor_9 sub1(clk,sub_temp1,underflow,{carry,temp1},1'b0);
//if there is a carry out => underflow
always@(*)
if(underflow)
begin 
sub_temp<=9'b000000001;
end
else
begin 
sub_temp<=sub_temp1;
end

and(overflow,sub_temp[8],1'b1); //if the exponent has more than 8 bits: overflow
//taking product of mantissa:
wire [47:0] prdt;
product p1(clk,prdt,{1'b1,inp1[22:0]},{1'b1,inp2[22:0]});
wire norm_flag;
wire [22:0] adj_mantissa;
normalize norm1(clk,adj_mantissa,norm_flag,prdt);
ripple_8 ripple_norm(clk,test_exp,dummy,sub_temp[7:0],{7'b0,norm_flag},1'b0);
assign out[31] = sign;
assign out[30:23] = test_exp;
assign out[22:0] = adj_mantissa;
endmodule


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//File Name: Multiplication.v
//Created By: Sheetal Swaroop Burada
//Date: 30-04-2019
//Project Name: Design of 32 Bit Floating Point ALU Based on Standard IEEE-754 in Verilog and its implementation on FPGA.
//University: Dayalbagh Educational Institute
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

module Multiplication(
    input clk,
		input [31:0] a_operand,
		input [31:0] b_operand,
		output reg Exception,Overflow,Underflow,
		output reg [31:0] result
		);



reg sign,product_round,normalised,zero;
reg [8:0] exponent,sum_exponent;
reg [22:0] product_mantissa;
reg [23:0] operand_a,operand_b;
reg [47:0] product,product_normalised; 

wire sign1,product_round1,normalised1,zero1;////
wire [8:0] exponent1,sum_exponent1;//
wire [22:0] product_mantissa1;
wire [23:0] operand_a1,operand_b1;//
wire [47:0] product1,product_normalised1;//
wire Exception1,Overflow1,Underflow1;//
wire [31:0] result1;
reg [31:0] res1,res2,res3;

assign sign1 = a_operand[31] ^ b_operand[31];

//Exception flag sets 1 if either one of the exponent is 255.
assign Exception1 = (&a_operand[30:23]) | (&b_operand[30:23]);

//Assigining significand values according to Hidden Bit.
//If exponent is equal to zero then hidden bit will be 0 for that respective significand else it will be 1

assign operand_a1 = (|a_operand[30:23]) ? {1'b1,a_operand[22:0]} : {1'b0,a_operand[22:0]};

assign operand_b1 = (|b_operand[30:23]) ? {1'b1,b_operand[22:0]} : {1'b0,b_operand[22:0]};

assign product1 = operand_a * operand_b;			//Calculating Product

assign product_round1 = |product_normalised[22:0];  //Ending 22 bits are OR'ed for rounding operation.

assign normalised1 = product[47] ? 1'b1 : 1'b0;	

assign product_normalised1 = normalised ? product : product << 1;	//Assigning Normalised value based on 48th bit

//Final Manitssa.
assign product_mantissa1 = product_normalised[46:24] + {21'b0,(product_normalised[23] & product_round)};

assign zero1 = Exception ? 1'b0 : (product_mantissa == 23'd0) ? 1'b1 : 1'b0;

assign sum_exponent1 = a_operand[30:23] + b_operand[30:23];

assign exponent1 = sum_exponent - 8'd127 + normalised;


assign Overflow1 = ((exponent[8] & !exponent[7]) & !zero) ; //If overall exponent is greater than 255 then Overflow condition.


//Exception Case when exponent reaches its maximu value that is 384.

//If sum of both exponents is less than 127 then Underflow condition.
assign Underflow1 = ((exponent[8] & exponent[7]) & !zero) ? 1'b1 : 1'b0; 


assign result1 = Exception ? 32'd0 : zero ? {sign,31'd0} : Overflow ? {sign,8'hFF,23'd0} : Underflow ? {sign,31'd0} : {sign,exponent[7:0],product_mantissa};
always@(posedge clk)
begin
  sign<=sign1;
  Exception<=Exception1;
  operand_a<=operand_a1;
  operand_b<=operand_b1;
  product<=product1;
  product_round<=product_round1;
  normalised<=normalised1;
  product_normalised<=product_normalised1;
  product_mantissa<=product_mantissa1;
  zero<=zero1;
  sum_exponent<=sum_exponent1;
  exponent<=exponent1;
  Overflow<=Overflow1;
  Underflow<=Underflow1;
  
  res1<=result1;
  res2<=res1;
  res3<=res2;
  result<=res3;
end


endmodule

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//File Name: Iteration.v
//Created By: Sheetal Swaroop Burada
//Date: 30-04-2019
//Project Name: Design of 32 Bit Floating Point ALU Based on Standard IEEE-754 in Verilog and its implementation on FPGA.
//University: Dayalbagh Educational Institute
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


module Iteration(
  input clk,
	input [31:0] operand_1,
	input [31:0] operand_2,
	output reg [31:0] solution
	);

wire [31:0] Intermediate_Value11,Intermediate_Value2;
wire [31:0] solution1;
reg [31:0] Intermediate_Value1;


controlx M1(clk,operand_1,operand_2,Intermediate_Value11,,);

//32'h4000_0000 -> 2.
Addition_Subtraction A1(clk,32'h4000_0000,{1'b1,Intermediate_Value1[30:0]},1'b0,,Intermediate_Value2);

controlx M2(clk,operand_1,Intermediate_Value2,solution1,,);
always@(posedge clk)
begin
  Intermediate_Value1<=Intermediate_Value11;
  solution<=solution1; 
end
endmodule

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//File Name: Additon_Subtraction.v
//Created By: Sheetal Swaroop Burada
//Date: 30-04-2019
//Project Name: Design of 32 Bit Floating Point ALU Based on Standard IEEE-754 in Verilog and its implementation on FPGA.
//University: Dayalbagh Educational Institute
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////




module Addition_Subtraction(
input clk,
input [31:0] a_operand,b_operand, //Inputs in the format of IEEE-754 Representation.
input AddBar_Sub,				  //If Add_Sub is low then Addition else Subtraction.
output reg Exception,
output reg [31:0] result              //Outputs in the format of IEEE-754 Representation.
);

reg operation_sub_addBar;
wire Comp_enable;
reg output_sign;
wire [31:0] operand_a,operand_b;
reg [23:0] significand_a,significand_b;
reg [7:0] exponent_diff;
reg [23:0] significand_b_add_sub;
reg [7:0] exponent_b_add_sub;
reg [24:0] significand_add;
wire [30:0] add_sum;
reg [23:0] significand_sub_complement;
reg [24:0] significand_sub;
wire [30:0] sub_diff;
reg [24:0] subtraction_diff; 
reg [7:0] exponent_sub;

wire operation_sub_addBar1;
wire Comp_enable1;
wire output_sign1;
wire [31:0] operand_a1,operand_b1;
wire [23:0] significand_a1,significand_b1;
wire [7:0] exponent_diff1;
wire [23:0] significand_b_add_sub1;
wire [7:0] exponent_b_add_sub1;
wire [24:0] significand_add1;
wire [30:0] add_sum1;
wire [23:0] significand_sub_complement1;
wire [24:0] significand_sub1;
wire [30:0] sub_diff1;
wire [24:0] subtraction_diff1; 
wire [7:0] exponent_sub1;
wire Exception1;
wire [31:0] result1;
wire perform;
wire [7:0] exp_a,exp_b;

//for operations always operand_a must not be less than b_operand
assign {Comp_enable,operand_a,operand_b} = (a_operand[30:0] < b_operand[30:0]) ? {1'b1,b_operand,a_operand} : {1'b0,a_operand,b_operand};


assign exp_a = operand_a[30:23];


assign exp_b = operand_b[30:23];


//Exception flag sets 1 if either one of the exponent is 255.
assign Exception1 = (&operand_a[30:23]) | (&operand_b[30:23]);

assign output_sign1 = AddBar_Sub ? Comp_enable ? !operand_a[31] : operand_a[31] : operand_a[31] ;

assign operation_sub_addBar1 = AddBar_Sub ? operand_a[31] ^ operand_b[31] : ~(operand_a[31] ^ operand_b[31]);

//Assigining significand values according to Hidden Bit.
//If exponent is equal to zero then hidden bit will be 0 for that respective significand else it will be 1
assign significand_a1 = (|operand_a[30:23]) ? {1'b1,operand_a[22:0]} : {1'b0,operand_a[22:0]};

assign significand_b1 = (|operand_b[30:23]) ? {1'b1,operand_b[22:0]} : {1'b0,operand_b[22:0]};

//Evaluating Exponent Difference
assign exponent_diff1 = operand_a[30:23] - operand_b[30:23];

//Shifting significand_b according to exponent_diff
assign significand_b_add_sub1 = significand_b >> exponent_diff;

assign exponent_b_add_sub1 = operand_b[30:23] + exponent_diff; 

//Checking exponents are same or not
assign perform = (operand_a[30:23] == exponent_b_add_sub);

///////////////////////////////////////////////////////////////////////////////////////////////////////
//------------------------------------------------ADD BLOCK------------------------------------------//

assign significand_add1 = (perform & operation_sub_addBar) ? (significand_a + significand_b_add_sub) : 25'd0; 

//Result will be equal to Most 23 bits if carry generates else it will be Least 22 bits.
assign add_sum[22:0] = significand_add[24] ? significand_add[23:1] : significand_add[22:0];


//If carry generates in sum value then exponent must be added with 1 else feed as it is.
assign add_sum[30:23] = significand_add[24] ? (1'b1 + operand_a[30:23]) : operand_a[30:23];


///////////////////////////////////////////////////////////////////////////////////////////////////////
//------------------------------------------------SUB BLOCK------------------------------------------//

assign significand_sub_complement1 = (perform & !operation_sub_addBar) ? ~(significand_b_add_sub) + 24'd1 : 24'd0 ; 

assign significand_sub1 = perform ? (significand_a + significand_sub_complement) : 25'd0;

priority_encoder pe(significand_sub,operand_a[30:23],subtraction_diff,exponent_sub);

assign sub_diff[30:23] = exponent_sub;


assign sub_diff[22:0] = subtraction_diff[22:0];


///////////////////////////////////////////////////////////////////////////////////////////////////////
//-------------------------------------------------OUTPUT--------------------------------------------//

//If there is no exception and operation will evaluate


assign result1 = Exception ? 32'b0 : ((!operation_sub_addBar) ? {output_sign,sub_diff} : {output_sign,add_sum});
always@(posedge clk)
begin  
  Exception<=Exception1;
  output_sign<=output_sign1;
  operation_sub_addBar<=operation_sub_addBar1;
  significand_a<=significand_a1;
  significand_b<=significand_b1;
  exponent_diff<=exponent_diff1;
  significand_b_add_sub<=significand_b_add_sub1;
  exponent_b_add_sub<=exponent_b_add_sub1;
  significand_add<=significand_add1;
  significand_sub_complement<=significand_sub_complement1;
  significand_sub<=significand_sub1;
  result<=result1;
end

endmodule

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//File Name: Priority Encoder.v
//Created By: Sheetal Swaroop Burada
//Date: 30-04-2019
//Project Name: Design of 32 Bit Floating Point ALU Based on Standard IEEE-754 in Verilog and its implementation on FPGA.
//University: Dayalbagh Educational Institute
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


module priority_encoder(
			input [24:0] significand,
			input [7:0] Exponent_a,
			output reg [24:0] Significand,
			output [7:0] Exponent_sub
			);

reg [4:0] shift;

always @(significand)
begin
	casex (significand)
		25'b1_1xxx_xxxx_xxxx_xxxx_xxxx_xxxx :	begin
													Significand = significand;
									 				shift = 5'd0;
								 			  	end
		25'b1_01xx_xxxx_xxxx_xxxx_xxxx_xxxx : 	begin						
										 			Significand = significand << 1;
									 				shift = 5'd1;
								 			  	end

		25'b1_001x_xxxx_xxxx_xxxx_xxxx_xxxx : 	begin						
										 			Significand = significand << 2;
									 				shift = 5'd2;
								 				end

		25'b1_0001_xxxx_xxxx_xxxx_xxxx_xxxx : 	begin 							
													Significand = significand << 3;
								 	 				shift = 5'd3;
								 				end

		25'b1_0000_1xxx_xxxx_xxxx_xxxx_xxxx : 	begin						
									 				Significand = significand << 4;
								 	 				shift = 5'd4;
								 				end

		25'b1_0000_01xx_xxxx_xxxx_xxxx_xxxx : 	begin						
									 				Significand = significand << 5;
								 	 				shift = 5'd5;
								 				end

		25'b1_0000_001x_xxxx_xxxx_xxxx_xxxx : 	begin						// 24'h020000
									 				Significand = significand << 6;
								 	 				shift = 5'd6;
								 				end

		25'b1_0000_0001_xxxx_xxxx_xxxx_xxxx : 	begin						// 24'h010000
									 				Significand = significand << 7;
								 	 				shift = 5'd7;
								 				end

		25'b1_0000_0000_1xxx_xxxx_xxxx_xxxx : 	begin						// 24'h008000
									 				Significand = significand << 8;
								 	 				shift = 5'd8;
								 				end

		25'b1_0000_0000_01xx_xxxx_xxxx_xxxx : 	begin						// 24'h004000
									 				Significand = significand << 9;
								 	 				shift = 5'd9;
								 				end

		25'b1_0000_0000_001x_xxxx_xxxx_xxxx : 	begin						// 24'h002000
									 				Significand = significand << 10;
								 	 				shift = 5'd10;
								 				end

		25'b1_0000_0000_0001_xxxx_xxxx_xxxx : 	begin						// 24'h001000
									 				Significand = significand << 11;
								 	 				shift = 5'd11;
								 				end

		25'b1_0000_0000_0000_1xxx_xxxx_xxxx : 	begin						// 24'h000800
									 				Significand = significand << 12;
								 	 				shift = 5'd12;
								 				end

		25'b1_0000_0000_0000_01xx_xxxx_xxxx : 	begin						// 24'h000400
									 				Significand = significand << 13;
								 	 				shift = 5'd13;
								 				end

		25'b1_0000_0000_0000_001x_xxxx_xxxx : 	begin						// 24'h000200
									 				Significand = significand << 14;
								 	 				shift = 5'd14;
								 				end

		25'b1_0000_0000_0000_0001_xxxx_xxxx  : 	begin						// 24'h000100
									 				Significand = significand << 15;
								 	 				shift = 5'd15;
								 				end

		25'b1_0000_0000_0000_0000_1xxx_xxxx : 	begin						// 24'h000080
									 				Significand = significand << 16;
								 	 				shift = 5'd16;
								 				end

		25'b1_0000_0000_0000_0000_01xx_xxxx : 	begin						// 24'h000040
											 		Significand = significand << 17;
										 	 		shift = 5'd17;
												end

		25'b1_0000_0000_0000_0000_001x_xxxx : 	begin						// 24'h000020
									 				Significand = significand << 18;
								 	 				shift = 5'd18;
								 				end

		25'b1_0000_0000_0000_0000_0001_xxxx : 	begin						// 24'h000010
									 				Significand = significand << 19;
								 	 				shift = 5'd19;
												end

		25'b1_0000_0000_0000_0000_0000_1xxx :	begin						// 24'h000008
									 				Significand = significand << 20;
								 					shift = 5'd20;
								 				end

		25'b1_0000_0000_0000_0000_0000_01xx : 	begin						// 24'h000004
									 				Significand = significand << 21;
								 	 				shift = 5'd21;
								 				end

		25'b1_0000_0000_0000_0000_0000_001x : 	begin						// 24'h000002
									 				Significand = significand << 22;
								 	 				shift = 5'd22;
								 				end

		25'b1_0000_0000_0000_0000_0000_0001 : 	begin						// 24'h000001
									 				Significand = significand << 23;
								 	 				shift = 5'd23;
								 				end

		25'b1_0000_0000_0000_0000_0000_0000 : 	begin						// 24'h000000
								 					Significand = significand << 24;
							 	 					shift = 5'd24;
								 				end
		default : 	begin
						Significand = (~significand) + 1'b1;
						shift = 8'd0;
					end

	endcase
end
assign Exponent_sub = Exponent_a - shift;

endmodule

// Code your design here
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//File Name: Division.v
//Created By: Sheetal Swaroop Burada
//Date: 30-04-2019
//Project Name: Design of 32 Bit Floating Point ALU Based on Standard IEEE-754 in Verilog and its implementation on FPGA.
//University: Dayalbagh Educational Institute
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


module Division(
  input clk,
	input [31:0] a_operand,
	input [31:0] b_operand,
	output Exception,
	output [31:0] result
);

wire sign1;
wire [7:0] shift1;
wire [7:0] exponent_a1;
wire [31:0] divisor1;
wire [31:0] operand_a1;
wire [31:0] Intermediate_X01;
wire [31:0] Iteration_X01;
wire [31:0] Iteration_X11;
wire [31:0] Iteration_X21;
wire [31:0] Iteration_X31;
wire [31:0] solution1;

wire [31:0] denominator1;
wire [31:0] operand_a_change1;

reg sign;//
reg [7:0] shift;//
reg [7:0] exponent_a;//
reg [31:0] divisor;//
reg [31:0] operand_a;//
reg [31:0] Intermediate_X0;//
reg [31:0] Iteration_X0;//
reg [31:0] Iteration_X1;//
reg [31:0] Iteration_X2;//
reg [31:0] Iteration_X3;//
reg [31:0] solution;//

reg [31:0] denominator;//
reg [31:0] operand_a_change;//

assign Exception = (&a_operand[30:23]) | (&b_operand[30:23]);

assign sign1 = a_operand[31] ^ b_operand[31];

assign shift1 = 8'd126 - b_operand[30:23];

assign divisor1 = {1'b0,8'd126,b_operand[22:0]};

assign denominator1 = divisor;

assign exponent_a1 = a_operand[30:23] + shift;

assign operand_a1 = {a_operand[31],exponent_a,a_operand[22:0]};

assign operand_a_change1 = operand_a;

//32'hC00B_4B4B = (-37)/17
controlx x0(clk,32'hC00B_4B4B,divisor,Intermediate_X01,,);

//32'h4034_B4B5 = 48/17
Addition_Subtraction X0(clk,Intermediate_X0,32'h4034_B4B5,1'b0,,Iteration_X01);

Iteration X1(clk,Iteration_X0,divisor,Iteration_X11);

Iteration X2(clk,Iteration_X1,divisor,Iteration_X21);

Iteration X3(clk,Iteration_X2,divisor,Iteration_X31);

controlx END(clk,Iteration_X3,operand_a,solution1,,);
always@(posedge clk)
begin
  sign<=sign1;
  shift<=shift1;
  divisor<=divisor1;
  denominator<=denominator1;
  exponent_a<=exponent_a1;
  operand_a<=operand_a1;
  operand_a_change<=operand_a_change1;
  Intermediate_X0<=Intermediate_X01;
  Iteration_X0<=Iteration_X01;
  Iteration_X1<=Iteration_X11;
  Iteration_X2<=Iteration_X21;
  Iteration_X3<=Iteration_X31;
  solution<=solution1;
end

assign result = {sign,solution[30:0]};


endmodule


module fpadd(  input [31:0]Number1, input [31:0]Number2,input clk, output [31:0]Result);
    wire reset=1'b0;
    reg    [31:0] Num_shift_80,Num_shift_pipe2_80; 
    reg    [7:0]  Larger_exp_80,Larger_exp_pipe1_80,Larger_exp_pipe2_80,Larger_exp_pipe3_80,Larger_exp_pipe4_80,Larger_exp_pipe5_80,Final_expo_80;
    reg    [22:0] Small_exp_mantissa_80,Small_exp_mantissa_pipe2_80,S_exp_mantissa_pipe2_80,S_exp_mantissa_pipe3_80,Small_exp_mantissa_pipe3_80;
    reg    [22:0] S_mantissa_80,L_mantissa_80;
    reg    [22:0] L1_mantissa_pipe2_80,L1_mantissa_pipe3_80,Large_mantissa_80,Final_mant_80;
    reg    [22:0] Large_mantissa_pipe2_80,Large_mantissa_pipe3_80,S_mantissa_pipe4_80,L_mantissa_pipe4_80;
    reg    [23:0] Add_mant_80,Add1_mant_80,Add_mant_pipe5_80;
    reg    [7:0]  e1_80,e1_pipe1_80,e1_pipe2_80,e1_pipe3_80,e1_pipe4_80,e1_pipe5_80;
    reg    [7:0]  e2_80,e2_pipe1_80,e2_pipe2_80,e2_pipe3_80,e2_pipe4_80,e2_pipe5_80;
    reg    [22:0] m1_80,m1_pipe1_80,m1_pipe2_80,m1_pipe3_80,m1_pipe4_80,m1_pipe5_80;
    reg    [22:0] m2_80,m2_pipe1_80,m2_pipe2_80,m2_pipe3_80,m2_pipe4_80,m2_pipe5_80;

    reg           s1_80,s2_80,Final_sign_80,s1_pipe1_80,s1_pipe2_80,s1_pipe3_80,s1_pipe4_80,s1_pipe5_80;
    reg           s2_pipe1_80,s2_pipe2_80,s2_pipe3_80,s2_pipe4_80,s2_pipe5_80;
    reg    [3:0]     renorm_shift_80,renorm_shift_pipe5_80;
    integer signed   renorm_exp_80;

    //:w
    //reg    [3:0]  renorm_exp_80,renorm_exp_pipe5_80;
    reg    [31:0] Result_80;

    assign Result = Result_80;

    always @(*) begin
        ///////////////////////// Combinational stage1 ///////////////////////////
	e1_80 = Number1[30:23];
	e2_80 = Number2[30:23];
        m1_80 = Number1[22:0];
	m2_80 = Number2[22:0];
	s1_80 = Number1[31];
	s2_80 = Number2[31];
        
        if (e1_80  > e2_80) begin
            Num_shift_80           = e1_80 - e2_80;              // determine number of mantissa shift
            Larger_exp_80           = e1_80;                     // store higher exponent
            Small_exp_mantissa_80  = m2_80;
            Large_mantissa_80      = m1_80;
        end 
        else begin
            Num_shift_80           = e2_80 - e1_80;
            Larger_exp_80           = e2_80;
            Small_exp_mantissa_80  = m1_80;
            Large_mantissa_80      = m2_80;
        end

	if (e1_80 == 0 | e2_80 ==0) begin
	    Num_shift_80 = 0;
	end
	else begin
	    Num_shift_80 = Num_shift_80;
	end	
        ///////////////////////// Combinational stage2 ///////////////////////////
        //right shift mantissa of smaller exponent
	if (e1_pipe2_80 != 0) begin
            S_exp_mantissa_pipe2_80  = {1'b1,Small_exp_mantissa_pipe2_80[22:1]};
	    S_exp_mantissa_pipe2_80  = (S_exp_mantissa_pipe2_80 >> Num_shift_pipe2_80);
        end
	else begin
	    S_exp_mantissa_pipe2_80 = Small_exp_mantissa_pipe2_80;
	end

	if (e2_80!= 0) begin
            L1_mantissa_pipe2_80      = {1'b1,Large_mantissa_pipe2_80[22:1]};
	end
	else begin
	    L1_mantissa_pipe2_80 = Large_mantissa_pipe2_80;
	end	
        ///////////////////////// Combinational stage3 ///////////////////////////
	//compare which is smaller mantissa
        if (S_exp_mantissa_pipe3_80  < L1_mantissa_pipe3_80) begin
                S_mantissa_80 = S_exp_mantissa_pipe3_80;
		L_mantissa_80 = L1_mantissa_pipe3_80;
        end
        else begin 
		S_mantissa_80 = L1_mantissa_pipe3_80;
		L_mantissa_80 = S_exp_mantissa_pipe3_80;
        end       
	///////////////////////// Combinational stage4 ///////////////////////////      
        //add the two mantissa's
	if (e1_pipe4_80!=0 & e2_pipe4_80!=0) begin
		if (s1_pipe4_80 == s2_pipe4_80) begin
        		Add_mant_80 = S_mantissa_pipe4_80 + L_mantissa_pipe4_80;
		end else begin
			Add_mant_80 = L_mantissa_pipe4_80 - S_mantissa_pipe4_80;
		end
	end	
	else begin
		Add_mant_80 = L_mantissa_pipe4_80;
	end      
	//determine shifts for renormalization for mantissa and exponent
	if (Add_mant_80[23]) begin
		renorm_shift_80 = 4'd1;
		renorm_exp_80 = 4'd1;
	end
	else if (Add_mant_80[22])begin
		renorm_shift_80 = 4'd2;
		renorm_exp_80 = 0;		
	end
	else if (Add_mant_80[21])begin
		renorm_shift_80 = 4'd3; 
		renorm_exp_80 = -1;
	end 
	else if (Add_mant_80[20])begin
		renorm_shift_80 = 4'd4; 
		renorm_exp_80 = -2;		
	end  
	else if (Add_mant_80[19])begin
		renorm_shift_80 = 4'd5; 
		renorm_exp_80 = -3;		
	end     
        else begin
		renorm_exp_80 = 0;
	end	
	///////////////////////// Combinational stage5 /////////////////////////////
	//Shift the mantissa as required; re-normalize exp; determine sign
        Final_expo_80 =  Larger_exp_pipe5_80 + renorm_exp_80;
	if (renorm_shift_pipe5_80 != 0) begin	
		Add1_mant_80 = Add_mant_pipe5_80 << renorm_shift_pipe5_80;
	end
	else begin
		Add1_mant_80 = Add_mant_pipe5_80;
	end
	Final_mant_80 = Add1_mant_80[23:1];  	      
	if (s1_pipe5_80 == s2_pipe5_80) begin
		Final_sign_80 = s1_pipe5_80;
	end 
	if (e1_pipe5_80 > e2_pipe5_80) begin
		Final_sign_80 = s1_pipe5_80;	
	end else if (e2_80 > e1_80) begin
		Final_sign_80 = s2_pipe5_80;
	end
	else begin
		if (m1_pipe5_80 > m2_pipe5_80) begin
			Final_sign_80 = s1_pipe5_80;		
		end else begin
			Final_sign_80 = s2_pipe5_80;
		end
	end	
	Result_80 = {Final_sign_80,Final_expo_80,Final_mant_80}; 
    end
    
    always @(posedge clk) begin
            if(reset) begin                           //reset all reg at reset signal
                s1_pipe2_80 <=   0;
		s2_pipe2_80 <=   0;
		e1_pipe2_80 <=   0;
		e2_pipe2_80 <=   0;	
		m1_pipe2_80 <=   0;
		m2_pipe2_80 <=   0;
		Larger_exp_pipe2_80 <=   0;
		//stage2
		Small_exp_mantissa_pipe2_80 <=   0;
	        Large_mantissa_pipe2_80     <=   0;
		Num_shift_pipe2_80          <=   0;
		s1_pipe3_80 <=   0;
		s2_pipe3_80 <=   0;
		e1_pipe3_80 <=   0;
		e2_pipe3_80 <=   0;	
		m1_pipe3_80 <=   0;
		m2_pipe3_80 <=   0;
		Larger_exp_pipe3_80 <=   0; 
		s1_pipe4_80 <=   0;
		s2_pipe4_80 <=   0;
		e1_pipe4_80 <=   0;
		e2_pipe4_80 <=   0;	
		m1_pipe4_80 <=   0;
		m2_pipe4_80 <=   0;
		Larger_exp_pipe4_80 <=  0; 
		s1_pipe5_80 <=   0;
		s2_pipe5_80 <=   0;
		e1_pipe5_80 <=   0;
		e2_pipe5_80 <=   0;	
		m1_pipe5_80 <=   0;
		m2_pipe5_80 <=   0;
		Larger_exp_pipe5_80 <= 0; 
		//stage3	
		S_exp_mantissa_pipe3_80  <= 0;
	       	L1_mantissa_pipe3_80     <= 0;
		//stage4
		S_mantissa_pipe4_80       <= 0;
		L_mantissa_pipe4_80       <= 0;	
		//stage5	
		Add_mant_pipe5_80 <= 0;
		renorm_shift_pipe5_80 <= 0;
		Result_80 <=0;
            end
	    else begin        
		///////////////////////////////PIPELINE STAGES and VARIABLES/////////////////
         	//propogate pipelined variables to next stages
		s1_pipe2_80 <=   s1_80;
		s2_pipe2_80 <=   s2_80;
		e1_pipe2_80 <=   e1_80;
		e2_pipe2_80 <=   e2_80;	
		m1_pipe2_80 <=   m1_80;
		m2_pipe2_80 <=   m2_80;
		Larger_exp_pipe2_80 <=   Larger_exp_80;
		//stage2
		Small_exp_mantissa_pipe2_80 <=   Small_exp_mantissa_80;
	        Large_mantissa_pipe2_80     <=   Large_mantissa_80;
		Num_shift_pipe2_80          <=   Num_shift_80;
		s1_pipe3_80 <=   s1_pipe2_80;
		s2_pipe3_80 <=   s2_pipe2_80;
		e1_pipe3_80 <=   e1_pipe2_80;
		e2_pipe3_80 <=   e2_pipe2_80;	
		m1_pipe3_80 <=   m1_pipe2_80;
		m2_pipe3_80 <=   m2_pipe2_80;
		Larger_exp_pipe3_80 <=   Larger_exp_pipe2_80; 
		s1_pipe4_80 <=   s1_pipe3_80;
		s2_pipe4_80 <=   s2_pipe3_80;
		e1_pipe4_80 <=   e1_pipe3_80;
		e2_pipe4_80 <=   e2_pipe3_80;	
		m1_pipe4_80 <=   m1_pipe3_80;
		m2_pipe4_80 <=   m2_pipe3_80;
		Larger_exp_pipe4_80 <=   Larger_exp_pipe3_80; 
		s1_pipe5_80 <=   s1_pipe4_80;
		s2_pipe5_80 <=   s2_pipe4_80;
		e1_pipe5_80 <=   e1_pipe4_80;
		e2_pipe5_80 <=   e2_pipe4_80;	
		m1_pipe5_80 <=   m1_pipe4_80;
		m2_pipe5_80 <=   m2_pipe4_80;
		Larger_exp_pipe5_80 <=   Larger_exp_pipe4_80; 
		//stage3	
		S_exp_mantissa_pipe3_80  <= S_exp_mantissa_pipe2_80;
	       	L1_mantissa_pipe3_80     <= L1_mantissa_pipe2_80;
		//stage4
		S_mantissa_pipe4_80         <=   S_mantissa_80;
		L_mantissa_pipe4_80         <=   L_mantissa_80;	
		//stage5	
		Add_mant_pipe5_80 <= Add_mant_80;
		renorm_shift_pipe5_80 <= renorm_shift_80;
		//renorm_exp_pipe5_80 <= renorm_exp_80;		
	   end
    end
    
endmodule



module comparatorx(
  input[31:0] a,
  input clk,
  input clkm,
  input reset,
  output comp);
  reg[31:0] neg_b,pos_b;
  reg out_temp;
  wire[31:0] out;
always@(posedge clk)begin
if(reset)begin
  neg_b <= 32'b00111100101000111101011100001010; //0.02
  pos_b <= 32'b10111100101000111101011100001010; //-0.02
end
else begin
end
end
fpadd add (a,neg_b,clkm,out);
always@(negedge clk)
    begin
      if(a==pos_b)begin
        out_temp <= 1'b1; //1 a>=b
      end
      else if (out[31] == 1'b1)begin
        out_temp <= 1'b0; //0 a<b
       
      end
      else if (out[31] == 1'b0)begin
       out_temp <= 1'b1; //1 a >=b
      end
      end
assign comp = out_temp;
endmodule

//Module for cubic
module square(input clk,reset,enable,input [31:0] a,output reg [31:0] out);

wire underflow1,overflow1,underflow2,overflow2,underflow3,overflow3;
wire [31:0] b,c,d;
reg [31:0] consta;

controlx CVQ1( clk,a,a,b,underflow1,overflow1);
controlx CVQ2( clk,b,a,c,underflow2,overflow2);
controlx CVQ3( clk,c,consta,d,underflow3,overflow3);

always@(posedge clk)
begin
  if(reset)
    begin
      consta<=32'b10111011010110100111001111111111;
    end
else if(enable)
    begin
      out<=d ;
    end
end
endmodule



// Module for w[n]
module SYNC1(
	clkm,clk, reset, enable,clkx,
	 del,
	 Prel,
	 A,
	D,
	 W
	
);

  input clkm,clk, reset, enable,clkx;
	input [31:0] del;
	input [31:0] Prel;
	input [31:0] A;
	input [31:0] D;
	output [31:0] W;
	
	reg [31:0] W;

wire[31:0] a,b,c,d;
wire underflow1,overflow1,underflow2,overflow2,underflow3,overflow3;
reg [31:0] ax,bx,cx;
integer counter;

controlx MM2(clkm,D,A,a,underflow1,overflow1);//D*A
controlx MM3(clkm,ax,Prel, b,underflow2,overflow2);//Prel*(D*A)
controlx MM4(clkm,bx,del,c,underflow3,overflow3);//del*(Prel*(D*A))
fpadd AA99(W,cx,clkm,d);//W+(del*(Prel*(D*A)))


always@(posedge clk)

begin
 if (reset) begin
	
	W<= 32'b00101011100011001011110011001100;
	ax<=32'b00101011100011001011110011001100;bx<=32'b00101011100011001011110011001100;cx<=32'b00101011100011001011110011001100;
	counter<=0;
 end 
 else if (enable) begin
	
	

	if(overflow1||underflow1)
	begin
	ax<=32'b00101011100011001011110011001100;
	end
	else begin
	ax<=a;end

	if(overflow2||underflow2)
	begin
	bx<=32'b00101011100011001011110011001100;
	end
	else begin
	bx<=b;end

	if(overflow3||underflow3)
	begin
	cx<=32'b00101011100011001011110011001100;
	end
	else begin
	cx<=c;end
	
	if(counter<98)
	begin 
	  counter<=counter+1;
	  W<=32'b00101011100011001011110011001100;
	  end
	else
	  begin
	   W <= d;
	   end

 end	
end

endmodule

// Module for D[n]
module Neuro(
	 clkm,clk1, reset, enable,clkx,td,
	C0,
	C1,
	D
);
  input clkm,clk1, reset, enable,clkx,td;
	input [31:0] C0;
	input [31:0] C1;
	output [31:0] D;
	
	reg [31:0] D;


reg [31:0] S,ax,bx,D1,D11;
reg [31:0] const1,const0_1;
wire [31:0] a,b,c,d;	

wire underflow1,overflow1,underflow2,overflow2;

controlx MM6(clkm,C0,D, a,underflow1,overflow1);//C0*D
controlx MM7(clkm,S,C1, b,underflow2,overflow2);//S*C1
fpadd AA2(ax,bx,clkm,c); //a+b

fpadd ADD1(S,const1,clkm,d);


always@(posedge clk1)
begin
 if (reset) begin
	D<= 32'b00101011100011001011110011001100;D1<= 32'b00101011100011001011110011001100;D11<= 32'b00101011100011001011110011001100;
	S<=32'b00101011100011001011110011001100;
	ax<=32'b00101011100011001011110011001100;bx<=32'b00101011100011001011110011001100;
  const1<=32'b00111111100000000000000000000000;
  const0_1<=32'b00111101110011001100110011001100;
 end 
 else if (enable) begin
	begin
		
	        if(td)
	        	begin
	        		 S<=d;
	        	end
		else
			begin
	        		 S<=32'b00101011100011001011110011001100;
	        	end
		
	end
	if(overflow1||underflow1)
	begin
	ax<=32'b00101011100011001011110011001100;
	end
	else begin
	ax<=a;end

	if(overflow2||underflow2)
	begin
  bx<=32'b00101011100011001011110011001100;
	end
	else begin
	bx<=b;end
  
  D <= c;
 end
end

endmodule



// Module for RMtr[n]
module RMtrace_ar(
	clkm,clkb, reset, enable,clkx,
	C0,
	 RM1,
	 RMtr
);

  input clkm,clkb;
  input reset;
  input enable;
  input clkx;
	input [31:0] C0;
	input [31:0] RM1;
	output [31:0] RMtr;
	
	reg [31:0] RMtr;


wire [31:0] a,b,c;
wire underflow1,overflow1,underflow2,overflow2;
reg [31:0] ax,cx;
reg [31:0] C1;
controlx MM1(clkm,C0,RMtr, a,underflow1,overflow1);//C0*RMtr
controlx MMMM1(clkm,C1,RM1, c,underflow2,overflow2);//C1*RM

fpadd AA1(ax,cx,clkm,b);//a+c



always@(posedge clkb)
begin
 if (reset) begin
	
	RMtr<=32'b00101011100011001011110011001100;
	ax<=32'b00101011100011001011110011001100;
	cx<=32'b00101011100011001011110011001100;
	C1<=32'b00111100001000111101011100001010;//0.01
 end 
 else if (enable) begin

	if(overflow1||underflow1)
	begin
   ax<=32'b00101011100011001011110011001100;
	end
	else begin
	 ax<=a;end

	if(overflow2||underflow2)
	begin
	 cx<=32'b00101011100011001011110011001100;
	end
	else begin
	 cx<=c;end

	 RMtr <= b;//RMtr=(C0*RMtr)+(C1*RM)
 end
end
endmodule


// Module for Isyn[n]
module SYNC2(
	 clkm,clk, reset, enable,clkx,
	 W,
	 Xn,
	 tp,
	 Isyn
	
);

  input clkm,clk, reset, enable,clkx;
	input [31:0] W;
	input [31:0] Xn;
	input tp;
	
	output [31:0] Isyn;
	

reg [31:0] Isyn;
wire [31:0] Xn,aq,bq,cq,dq,eq,fq,gq,hq,iq,Sq,S1q,S11q,k0q;

wire underflow1,overflow1,underflow2,overflow2,underflow3,overflow3,underflow4,overflow4,underflow5,overflow5;
wire [31:0] a,b,c,d,e,f,g,h,i,j,z;
reg [31:0] dx_actual,fx_actual;
reg [31:0] const1;
reg [31:0] k0,S,S1,S11,ax,fx,gx,dx,jx;
reg [31:0] CONST;

fpadd AA99(W,k0,clkm,c); //W+k0
controlx MM99(clkm,Xn,c,d,underflow2,overflow2);//(W+k0)K

fpadd AA98(S,S11,clkm,e); 
controlx MM98(clkm,S,CONST,f,underflow3,overflow3);//S*time const
controlx MM97(clkm,fx,dx,g,underflow4,overflow4);//Isyn


fpadd ADD2(S,const1,clkm,h);
fpadd ADD8(S1,const1,clkm,i);

//fpadd ADD811(SS,gx,clkx,z);


assign aq=ax;
assign bq=b;
assign cq=c;
assign dq=dx;
assign eq=e;
assign fq=fx;
assign gq=gx;
assign hq=h;
assign iq=i;
assign Sq=S;
assign S1q=S1;
assign S11q=S11;
assign k0q=k0;

/*always@(posedge tp)
begin
 if(reset) begin
	S1=32'b00101011100011001011110011001100;
 end
 else if(enable) begin
	S1=i;
 end
end
*/

always@(posedge clk)
begin
	
	

 if (reset) begin
	Isyn<=32'b00101011100011001011110011001100;
	S<=32'b00101011100011001011110011001100;
	k0<=32'b00101011100011001011110011001100;
	
	ax<=32'b00101011100011001011110011001100;dx<=32'b00101011100011001011110011001100;
	fx<=32'b00101011100011001011110011001100;
	gx<=32'b00101011100011001011110011001100;
	jx<=32'b00101011100011001011110011001100;dx_actual<=32'b00101011100011001011110011001100;fx_actual<=32'b00101011100011001011110011001100;
	const1<=32'b00111111100000000000000000000000;
	CONST<=32'b00111010100000110001001001101110;//0.001
 end 
 else if (enable) begin	

	if(overflow1|underflow1)
	begin
	ax<=32'b00101011100011001011110011001100;
	end
	else begin
	ax<=a;end
	
	if(overflow2|underflow2)
	begin
	dx<=32'b00101011100011001011110011001100;
	end
	else begin
	dx<=d;end

	if(overflow3|underflow3)
	begin
	fx<=32'b00101011100011001011110011001100;
	end
	else begin
	fx<=f;
	end
	
	if(overflow4|underflow4)
	begin
	gx<=32'b00101011100011001011110011001100;
	end
	else begin
	gx<=g;end

	if(overflow5|underflow5)
	begin
	jx<=32'b00101011100011001011110011001100;
	end
	else begin
	jx<=j;end

	begin
			if(tp)
				begin
				S<=32'b00111111100000000000000000000000;
				k0<=Isyn;
				begin
				Isyn<=gx;
				end
				end
			else
				begin
				Isyn<=gx;
				k0<=k0;
				S<=h;
				end
			
		end
 end
end
endmodule

// Module for A[n]
module ActSpike(
	clkm,clkb, reset, enable,clkx,
	 C0,
	 del,
	 in1,in2,in3,
	tp,
	 out,
	 A1
);

  input clkm,clkb, reset, enable,clkx;
	input [31:0] C0;
	input [31:0] del;
	input [31:0] in1,in2,in3;
	input tp;
	output  [31:0] out;
	output  [31:0] A1;
	
	reg [31:0] A1;
	reg[31:0] out;

wire overflow1,overflow2,underflow1,underflow2,underflow3,overflow3;
reg [1:0]sel;
reg [31:0] S,f,g,ax,AA;
reg [31:0] const1,const0_1;
reg [31:0] consta;
wire[31:0] a,b,c,d,e;
wire h1;
reg h;



//comparator1 CCCN(C,consta,clkx,h1);//checking if a>b

controlx MM11(clkm,C0,AA,a,underflow3,overflow3);//C0*A
controlx MM12(clkm,out,S,b,underflow1,overflow1);//capital_del*S
controlx MM13(clkm,f,del,c,underflow2,overflow2);//b*del
fpadd AA3(ax,g,clkm,d);//a+c 

fpadd ADD4(S,const1,clkm,e);




always@(posedge clkb)
begin


	begin
		//if(tp&h)
		if(tp)
		sel<=2'b00;
		else //if(tp^h)
		sel<=2'b10;
		//else
		//sel<=2'b01;
	end

		case(sel)
			2'b00: out<=in1;//1
			2'b01: out<=in2;//0
			2'b10: out<=in3;//-1
		  2'b11: out<=in2;//0
		endcase
 if (reset) begin
	
	AA <= 32'b00101011100011001011110011001100;A1 <= 32'b00101011100011001011110011001100;
	S<=32'b00101011100011001011110011001100;f<=32'b00101011100011001011110011001100;g<=32'b00101011100011001011110011001100;ax<=32'b00101011100011001011110011001100;
	const1<=32'b00111111100000000000000000000000;
	consta<=32'b00110110011111011101111101110001;
	const0_1<=32'b00111101110011001100110011001100;
	h<=1'b1;
	
 end 
 else if (enable) begin
	
		
		 begin
			if(tp)
				begin
					 S<=e;
				end
			else
				begin
					 S<=32'b00101011100011001011110011001100;
				end
		 end
	

	if(overflow1||underflow1)
	begin
	 f<=32'b00101011100011001011110011001100;
	end
	else begin
	 f<=b;end
	
	if(overflow2||underflow2)
	begin
	g<=32'b00101011100011001011110011001100;
	end
	else begin
	g<=c;end
	
	if(overflow3||underflow3)
	begin
	ax<=32'b00101011100011001011110011001100;
	end
	else begin
  ax<=a;end

	AA<=d;
	
	if(AA[31]==1)
	begin
	A1<={~AA[31],AA[30:0]};
	end

	else
	begin
	A1<=AA;
	end
 end
end
endmodule

// Module for Prel[n]
module Prel(
	clkm,clk, reset, enable,clkx,clkxy,
	 Inh1,
  Prel,Pxy
);

  input clkm,clk, reset, enable,clkx,clkxy;
	input [31:0] Inh1;
	output [31:0] Prel;
	output [31:0] Pxy;
	
	reg [31:0] Prel;
	wire [31:0] aq,bq,cq,dq,eq,iq,jq,kq,mq,nq,n1q,pq,qq,uq,u1q,vq,Pxy;
	

reg [31:0] xn;
reg [31:0] y;
reg [31:0] C0;
//reg [31:0] const6 = 32'b01000000110000000000000000000000;
//reg [31:0] const4 = 32'b01000000100000000000000000000000 ;
reg [31:0] const1 ;
reg [31:0] constm1;
//reg[ 31:0] const10=32'b01000100011110100000000000000000;
reg[ 31:0] const10;
reg [31:0] const025;
reg [31:0] const1001;
reg [31:0] constm011;
reg [31:0] constm001;
reg [31:0] conste8;
reg [31:0] conste16;
reg [31:0] constm_00001;
reg [31:0] constm_00007;
reg [31:0] const1_1;
reg [31:0] const1000;
//reg [31:0] constm001=32'b10111010100000110001001001101110;

wire [31:0] u1;

wire [31:0] a,b,c,d,e,i,j,k,m,n,p,q,r,u,v,w,o,z,con1,con2,o1,Inh,k1;
wire overflow1,overflow2,underflow1,underflow2,underflow3,overflow3,overflow4,overflow5,underflow4,underflow5,underflow6,overflow6,underflow7,overflow7,underflow8,overflow8,underflow9,overflow9;
reg [31:0] ax,bx,cx,dx,ex,px,kx,m1,n1,ox,zx,nx,kkx;
//reg [31:0] const0=32'b00101011100011001011110011001100;
wire compare;
assign aq=ax;
assign bq=bx;
assign cq=cx;
assign dq=dx;
assign eq=ex;
assign iq=i;
assign jq=j;
assign kq=kx;
assign mq=m;
assign nq=n;
assign n1q=n1;
assign pq=px;
assign qq=q;
assign uq=u;
assign u1q=u1;
assign vq=v;
assign Pxy=n;


fpadd AddInh(Inh1,constm_00001,clkm,Inh);//Inh

controlx MM17xx(clkm,const1000,Inh,a,underflow1,overflow1);
controlx MM18xx(clkm,k1,const1000,k,underflow2,overflow2);
//controlx MM20xx(clkm,bx,i,c,underflow3,overflow3);//((10*Inh)-1)^3//

//fpadd AA5xx(constm1,ax,clkm,i);//(10*Inh)-1;//
//fpadd AA6xx(const1,bx,clkm,j);//1+square//
//fpadd AA7xx(i,cx,clkm,u);//linear+cubic//

//fpadd SS3xx(j,u1,clkm,v);//xn/10//

//controlx MM77xx(clkm,const10,v,k,underflow6,overflow6);//xn//


//controlx MM22xx(clkm,kx,dx,e,underflow5,overflow5);//x3

//controlx MM21xsssx(clkm,dx,conste8,o,underflow4,overflow4);//x2				xx
//controlx MM22xsssx(clkm,ex,conste16,z,underflow5,overflow5);//x3				xx

controlx MM21xx(clkm,kkx,constm_00007,d,underflow4,overflow4);
fpadd AA444xx(dx,const1_1,clkm,n);//       {Pinh}

//fpadd AA445xx(o1,m,clkx,n);

fpadd SS99xx(const1,n1,clkm,q);//1-Pinh

wire res1,res2;

Division DUT(clkm,const1,ax,Exception,k1);

controlx D1xx(clkm,q,const025,p,underflow7,overflow7);//0.25(1-Pinh)

comparatorx VFFF1(px,clkx,clkm,reset,res1);
//comparator2 VDG2(Inh,clkm,reset,res2,con2);

	assign u1={~u[31],u[30:0]};
	assign o1={~ox[31],ox[30:0]};
	assign compare=res1&&px[31];



always@(posedge clkx)
begin  
  
  
  
	  if(n[31]==1'b0)
	begin
	  n1<={~n[31],n[30:0]};
	end
 	 else if (n[31]==1'b1)
	begin
	 n1<=32'b00101011100011001011110011001100;
	end
end

always@(posedge clk)
begin

  





 if (reset) begin
	Prel<=32'b00101011100011001011110011001100;
	xn<=32'b00101011100011001011110011001100;
	y<=32'b00101011100011001011110011001100;
	ax<=32'b00101011100011001011110011001100;bx<=32'b00101011100011001011110011001100;cx<=32'b00101011100011001011110011001100;
	dx<=32'b00101011100011001011110011001100;ex<=32'b00101011100011001011110011001100;px<=32'b00101011100011001011110011001100;
	ox<=32'b00101011100011001011110011001100;zx<=32'b00101011100011001011110011001100;nx<=32'b00101011100011001011110011001100;
	kkx<=32'b00101011100011001011110011001100;
	C0 <= 32'b00111111100000000000000000000000;
  const1<= 32'b00111111100000000000000000000000;
  constm1<= 32'b10111111100000000000000000000000;
	const025 <= 32'b00111110100000000000000000000000;
  const1001<=32'b00111111100000000010000011000100;
  constm011<=32'b10111100001101000011100101011000;
  constm001<=32'b10111010100000110001001001101110;
  conste8<=32'b00110010001010111100110001110111;
  conste16<=32'b00100100111001101001010110010100;
  constm_00001<=32'b10110111001001111100010110101100;
  constm_00007<=32'b10111000100100101100110011110110;
  const1_1<=32'b00111111100011001100110011001100;
  const1000<=32'b01000100011110100000000000000000;
	
 end 
 
 else if (enable) begin
	if(overflow1||underflow1)
	begin
	ax<=32'b00101011100011001011110011001100;
	end
	else begin
	ax<=a;end
	
	if(overflow2||underflow2)
	begin
	kkx<=32'b00101011100011001011110011001100;
	end
	else begin
	kkx<=k;end
	
	if(overflow3||underflow3)
	begin
	cx<=32'b00101011100011001011110011001100;
	end
	else begin
	cx<=c;end

	if(overflow4||underflow4)
	begin
	dx<=32'b00101011100011001011110011001100;
	end
	else begin
	dx<=d;end
	
	if(overflow5||underflow5)
	begin
	ex<=32'b00101011100011001011110011001100;
	end
	else begin
	ex<=e;end

	if(Exception)
	begin
	kx<=32'b00101011100011001011110011001100;
	end
	else begin
	kx<=k;end
	

	if(overflow7||underflow7)
	begin
	px<=32'b00101011100011001011110011001100;
	end
	else begin
	px<=p;end

	if(overflow8||underflow8)
	begin
	ox<=32'b00101011100011001011110011001100;
	end
	else begin
	ox<=o;end
	

	if(overflow9||underflow9)
	begin
	 zx<=32'b00101011100011001011110011001100;
	end
	else begin
	 zx<=z;end

  if(res1&&px[31])
    begin
    Prel<=32'b00111000110100011011011100010111;
    end
  else if(px[31]==1'b1)
   begin
    Prel<=32'b00111110100000000000000000000000;
    end
  else
    begin
    Prel<=px;
  end
	  
  //Prel<=nx;  



 end
end
endmodule

// Module for Inh[n]
module Inh(
	clkm,clk, reset, enable,clkx,
	C0,
	C1,
	 RMtr,
	A,
	Inh
);

  input clkm,clk, reset, enable,clkx;
	input [31:0] C0;
	input [31:0]C1;
	input [31:0] RMtr;
	input [31:0] A;
	output[31:0] Inh;
	

wire[31:0] aq,bq,cq,dq,eq,fq,gq;
reg [31:0] Inh;

wire [31:0] a,b,c,d;
reg [31:0] ax,bx,cx;
wire underflow1,overflow1,underflow2,overflow2,underflow3,overflow3;

controlx MM14(clkm,RMtr,A,a,underflow1,overflow1);//RMtr*A
controlx MM15(clkm,ax,C1,b,underflow2,overflow2);//del*(RMtr*A)
controlx MM16(clkm,C0,Inh,c,underflow3,overflow3);//C0*Inh
fpadd AA4(bx,cx,clkm,d);//b+c

assign aq=ax;
assign bq=bx;
assign cq=cx;
assign dq=d;
assign eq=a;
assign fq=b;
assign gq=c;

always@(posedge clk)

begin
 if (reset) begin
	Inh <= 32'b00110111001001111100010110101100;
	ax<=32'b00101011100011001011110011001100;bx<=32'b00101011100011001011110011001100;cx<=32'b00101011100011001011110011001100;
 end 
 else if (enable) begin
	if(overflow1||underflow1)
	begin
	ax<=32'b00101011100011001011110011001100;
	end
	else begin
	ax<=a;end
	
	if(overflow2||underflow2)
	begin
	bx<=32'b00101011100011001011110011001100;
	end
	else begin
	bx<=b;end

	if(overflow3||underflow3)
	begin
	cx<=32'b00101011100011001011110011001100;
	end
	else begin
	cx<=c;end

	Inh<=d;

 end
end
endmodule

// Module for Y[n]
module Y(
	clkm,clk, reset, enable,clkx,
	tp,Yn
);

  input clkm,clk, reset, enable,clkx,tp;
  output[31:0] Yn;
	

reg [31:0] Yn;

reg[31:0] const0_9,const1,consty,const0_1;
wire [31:0] a,b,c,d;
reg [31:0] ax,bx,S;
wire underflow1,overflow1,underflow2,overflow2,underflow3,overflow3;

controlx YY14(clkm,const0_9,Yn,a,underflow1,overflow1);//0.9*Y
//fpadd YYYY1(S,const0_1,clkx,d);//S+0.1
controlx YY15(clkm,S,consty,b,underflow2,overflow2);//1*delta


fpadd YYYY3(bx,ax,clkm,c);

assign Sy=S;

always@(posedge clk)

begin
 if (reset) begin
	Yn <= 32'b00111111100000000000000000000000;
	const0_9<=32'b00111111011001100110011001100110;//0.9
	const1<=32'b00111111100000000000000000000000;
	const0_1<=32'b00111101110011001100110011001100;
	S<= 32'b00101011100011001011110011001100;
	ax<=32'b00101011100011001011110011001100;
  bx<=32'b00101011100011001011110011001100;
  consty<=32'b00111111100000000000000000000000;
 end 
 
 else if (enable) begin
   
   if(tp)
	begin
	 	 S<=const1;
	end

	else
	begin
	 S<=32'b00101011100011001011110011001100;
	 //S<=S;
	 
	 
	end
  
   
	if(overflow1||underflow1)
	begin
	ax<=32'b00101011100011001011110011001100;
	end
	else begin
	ax<=a;end
	
	if(overflow2||underflow2)
	begin
	bx<=32'b00101011100011001011110011001100;
	end
	else begin
	bx<=b;end
	
	
	Yn<=c;

 end
end
endmodule

// Module for X[n]
module X(
	clkm,clk, reset, enable,clkx,
	tp,Yn,Xn
);

  input clkm,clk, reset, enable,clkx,tp;
  input[31:0] Yn;
  output [31:0] Xn;
	

reg [31:0] Xn;

reg[31:0] const0_9,const0_1;
wire [31:0] a,b,c;
reg [31:0] ax,bx;
wire underflow1,overflow1,underflow2,overflow2,underflow3,overflow3;

controlx XX14(clkm,const0_9,Xn,a,underflow1,overflow1);
controlx XX15(clkm,const0_1,Yn,b,underflow2,overflow2);
fpadd XRREEX4(bx,ax,clkm,c);

assign axx=ax;
assign bxx=bx;
assign cxx=c;

always@(posedge clk)

begin
 if (reset) begin
	Xn <= 32'b00101011100011001011110011001100;
	const0_9<=32'b00111111011001100110011001100110;//0.9
	const0_1<=32'b00111101110011001100110011001100;//0.1
	ax<=32'b00101011100011001011110011001100;
	bx<=32'b00101011100011001011110011001100;
 end 
 
 else if (enable) begin
   
	if(overflow1||underflow1)
	begin
	ax<=32'b00101011100011001011110011001100;
	end
	else begin
	ax<=a;end
	
	
	if(overflow2||underflow2)
	begin
	bx<=32'b00101011100011001011110011001100;
	end
	else begin
	bx<=b;end
	
	
	Xn<=c;

 end
end
endmodule


module PreSynapse(input clkm,clkxy,clkb, reset1, enable1,clkx1,
	input tp1,
	
	input clkd, reset2, enable2,clkx2,

	
	
	input clkc, reset3, enable3,clkx3,
	input [31:0] RM1,

	input clka, reset4, enable4,clkx4,td4,


	input clkp, reset6, enable6,clkxp,

	input clk5, reset5, enable5,clkx5,
	
	

	input clk8, reset8, enable8,clkx7,
	

	input clks, reset7, enable7,clkx8,tp2,
	output [31:0] Isyn7,WWx,Ax,Dx,RMtrx,Prelx,Inhx,Pxy,Xn,Yn
);
wire [31:0] out1;



reg [31:0] C01;//
reg [31:0] del1;//
reg[31:0]in1;//1
reg[31:0]in2;//0
reg[31:0]in3;//-1

reg [31:0] C02;//
reg [31:0]C12;//



reg [31:0] C04;//
reg[31:0] C14;//

reg [31:0] del5;//

reg [31:0] C08;//

reg [31:0] C03;//


wire [31:0] ACTtoINHtoSYNC1,RMTtoINH,INHtoPREL,SYNC1toPREL,NEUROtoSYNC1,StoS2,KtoSYNC2;

assign WWx=StoS2;
assign Ax=ACTtoINHtoSYNC1;
assign Dx=NEUROtoSYNC1;
assign RMtrx=RMTtoINH;
assign Prelx=SYNC1toPREL;
assign KKx=KtoSYNC2;
assign Inhx=INHtoPREL;

ActSpike AC1(clkm,clkb,reset1,enable1,clkx1,C01,del1,in1,in2,in3,tp1,out1,ACTtoINHtoSYNC1);
RMtrace_ar RMtr(clkm,clkc,reset3,enable3,clkx3,C03,RM1,RMTtoINH);
Inh inh1(clkm,clkd,reset2,enable2,clkx2,C02,C12,RMTtoINH,ACTtoINHtoSYNC1,INHtoPREL);


Neuro N(clkm,clka,reset4,enable4,clkx4,td4,C04,C14,NEUROtoSYNC1);
Prel p(clkm,clkp,reset6,enable6,clkxp,clkxy,INHtoPREL,SYNC1toPREL,Pxy);


SYNC1 S(clkm,clk5,reset5,enable5,clkx5,del5,SYNC1toPREL,ACTtoINHtoSYNC1,NEUROtoSYNC1,StoS2);


SYNC2 S2(clkm,clks,reset7,enable7,clkx7,StoS2,Xn,tp2,Isyn7);



Y Yn1(clkm,clkb,reset8,enable8,clkx8,tp1,Yn);
X Xn1(clkm,clkb,reset8,enable8,clkx8,tp1,Yn,Xn);

always@(posedge clka)begin
if(reset2)begin
  
C01<=32'b00111111011001100110011001100110;//0.9
del1<=32'b00111101110011001100110011001100;//0.1
in1<=32'b00111111100000000000000000000000;//1
in2<=32'b00101011100011001011110011001100;//0
in3<=32'b10111111100000000000000000000000;//-1

C02<=32'b00111111011001100110011001100110;//0.9
C12<=32'b00111100001000111101011100001010;//0.01

C04<=32'b00111111000000000000000000000000;//0.5
C14<=32'b00111111000000000000000000000000;//0.5

del5<=32'b00111100001000111101011100001010;//0.01

C08<=32'b00111101110011001100110011001100;//

C03<=32'b00111111011110011001100110011001;//0.975

    end
else
    begin
    end
end


endmodule 


// Module for S[n]
module S(
	input clk, reset, enable,clkm,
	//input [31:0] K0,K1,K2,K3,K4,
	input [31:0] Isyn1,
	input tp,
	output reg [31:0] S,
	output [31:0] tanhxs,Isyns,xs,ws,ys,zs,ps,qs,rs
);

wire [31:0] x,a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,y,z,aa,bb,cc,dd,jj;
wire ee,ff,gg,hh,ii;
reg [31:0] tanhx,aa1,ax,cx,dx,ex,hx,ix,mx,nx,qx,rx,sx,tx,zx,aax,ccx,Isyn,jjx;
integer flag;
wire [31:0] S1;
reg [31:0] const100;//
reg [31:0] const1m;
reg [31:0] consta1;
reg [31:0] constb1;
reg [31:0] constc1;
reg [31:0] consta2;
reg [31:0] constb2;
reg [31:0] constc2;
reg [31:0] consta3;
reg [31:0] constb3;
reg [31:0] constc3;
reg [31:0] const2by15;
reg [31:0] const1by3m;
reg [31:0] const1;
reg [31:0] constm1;
reg [31:0] const1byDs;//
reg [31:0] const05;
reg [31:0] const15;
reg [31:0] const25;
reg [31:0] const4;
reg [31:0] constx;
reg [31:0] prev;
reg [31:0] conste14;
reg [31:0] const7;
reg [31:0] consty;

wire underflow1,overflow1,underflow2,overflow2,underflow3,overflow3,underflow4,overflow4,underflow5,overflow5,underflow6,overflow6,underflow7,overflow7,underflow8,overflow8,underflow9,overflow9,underflow10,overflow10,underflow11,overflow11,underflow12,overflow12,underflow13,overflow13,underflow14,overflow14,underflow15,overflow15,underflow16,overflow16;


controlx MMMN1(clkm,const100,Isyn1,a,underflow1,overflow1);
fpadd AAAN1(ax,const7,clkm,b);//100Isyn-1		{x}

controlx MMMN2(clkm,b,b,c,underflow2,overflow2);//x*x


controlx MMMN3(clkm,cx,constc1,d,underflow3,overflow3);
controlx MMMN4(clkm,b,constb1,e,underflow4,overflow4);
fpadd AAAN2(consta1,dx,clkm,f);
fpadd AAAN3(f,ex,clkm,g);//F(x)1

controlx MMMN5(clkm,cx,constc2,h,underflow5,overflow5);
controlx MMMN6(clkm,b,constb2,i,underflow6,overflow6);
fpadd AAAN4(consta2,hx,clkm,j);
fpadd AAAN5(j,ix,clkm,k);//F(x)2

controlx MMMN7(clkm,c,constc3,m,underflow7,overflow7);
controlx MMMN8(clkm,b,constb3,n,underflow8,overflow8);
fpadd AAAN6(consta2,mx,clkm,o);
fpadd AAAN7(o,nx,clkm,p);//F(x)3

controlx MMMN9(clkm,cx,b,q,underflow9,overflow9);//x*x*x
controlx MMMN10(clkm,cx,qx,r,underflow10,overflow10);//x*x*x*x*x
controlx MMMN11(clkm,rx,const2by15,s,underflow11,overflow11);//2/15x5
controlx MMMN12(clkm,qx,const1by3m,t,underflow12,overflow12);//-1/3x3
fpadd AAAN8(sx,tx,clkm,u);//x5+x3
fpadd AAAN9(u,b,clkm,v);//F(x)0


fpadd AAAN10(const1,tanhx,clkm,w);//1+tanhx
//fpadd AAAN11(const1,S1,clkm,y);//1-Sm
controlx MMMN13(clkm,w,constx,z,underflow13,overflow13);//2nd term

assign ps=w;
assign qs=zx;
assign rs=dd;

controlx MMMN14(clkm,S,consty,aa,underflow14,overflow14);//1st term
fpadd AAAN12(zx,aax,clkm,dd);//S[n+1]
//controlx MMMN15(clkm,bb,constx,cc,underflow15,overflow15);

assign ws=aax;
assign ys=bb;
assign zs=ccx;

//fpadd AAAN13(S,ccx,clkm,dd);

//comparator1 CCN1(b,const05,clkx,ee);
//comparator1 CCN2(b,const15,clkx,ff);
//comparator1 CCN3(b,const25,clkx,gg);
//comparator1 CCN4(b,const4,clkx,hh);
//comparator1 CCN5(prev,S,clkx,ii);

controlx Amplifier1(clkm,Isyn1,conste14,jj,underflow16,overflow16);

	assign tanhxs=tanhx;
	assign Isyns=Isyn;
	assign xs=b;

reg [31:0] S_next;

//assign S=S_next;
assign S1={~S[31],S[30:0]};  
  
always@(posedge clkm)
begin
   
	 Isyn<=jjx;
	 
end


always@(posedge clk)
begin	

 if (reset) begin
	S<=32'b00110000100010010111000001011111;
	//S_next<=32'b00110000100010010111000001011111;
	
	ax<=32'b00100010000000000000000000000001;
	cx<=32'b00100010000000000000000000000001;
	dx<=32'b00100010000000000000000000000001;
	ex<=32'b00100010000000000000000000000001;
	hx<=32'b00100010000000000000000000000001;
	ix<=32'b00100010000000000000000000000001;
	mx<=32'b00100010000000000000000000000001;
	nx<=32'b00100010000000000000000000000001;
	qx<=32'b00100010000000000000000000000001;
	rx<=32'b00100010000000000000000000000001;
	sx<=32'b00100010000000000000000000000001;
	tx<=32'b00100010000000000000000000000001;
	zx<=32'b00100010000000000000000000000001;
	aax<=32'b00100010000000000000000000000001;
	ccx<=32'b00100010000000000000000000000001;
	tanhx<=32'b00100010000000000000000000000001;
	
 const100<=32'b01000110000001001101000000000000;
 const1m<=32'b10111111100000000000000000000000;
 consta1<=32'b00111110010100100111100000111000;
 constb1<=32'b00111111001100010011000000011110;
 constc1<=32'b0111110000010110101100101011101;
 consta2<=32'b00111111011000100100110011111011;
 constb2<=32'b00111101101101110010011100001111;
 constc2<=32'b10111100000110011101011010111001;
 consta3<=32'b00111111011111000001010100110111;
 constb3<=32'b00111100001100011011001101110110;
 constc3<=32'b10111010000000000111001001110000;
 const2by15<=32'b00111110000010001000011110101000;
 const1by3m<=32'b10111110101010100111111011111001;
 const1<=32'b00111111100000000000000000000000;
 constm1<=32'b10111111100000000000000000000000;
 const1byDs<=32'b11001101100011001101010010111100;//
 const05<=32'b00111111000000000000000000000000;
 const15<=32'b00111111110000000000000000000000;
 const25<=32'b01000000001000000000000000000000;
 const4<=32'b01000000100000000000000000000000;
 constx<=32'b00110000100010010111000001011111;//1e-9
 prev<=32'b00100010000000000000000000000001;
 conste14<=32'b01010110101101011110011000100000;
 const7<=32'b10111111010110011001100110011001;
 flag<=0;
 consty<=32'b00111111001101000110010010011001;//0.70466
 
 end 
 else if (enable) begin
	
	
	if(overflow1||underflow1)
	begin
	 ax<=32'b00100010000000000000000000000001;
	end
	else begin
	 ax<=a;end

	if(overflow2||underflow2)
	begin
	 cx<=32'b00100010000000000000000000000001;
	end
	else begin
	 cx<=c;end	
	
	if(overflow3||underflow3)
	begin
	 dx<=32'b00100010000000000000000000000001;
	end
	else begin
	 dx<=d;end

	if(overflow4||underflow4)
	begin
	 ex<=32'b00100010000000000000000000000001;
	end
	else begin
	 ex<=e;end

	if(overflow5||underflow5)
	begin
	 hx<=32'b00100010000000000000000000000001;
	end
	else begin
	 hx<=h;end

	if(overflow6||underflow6)
	begin
	 ix<=32'b00100010000000000000000000000001;
	end
	else begin
	 ix<=i;end

	if(overflow7||underflow7)
	begin
	 mx<=32'b00100010000000000000000000000001;
	end
	else begin
	 mx<=m;end

	if(overflow8||underflow8)
	begin
	 nx<=32'b00100010000000000000000000000001;
	end
	else begin
	 nx<=n;end

	if(overflow9||underflow9)
	begin
	 qx<=32'b00100010000000000000000000000001;
	end
	else begin
	 qx<=q;end

	if(overflow10||underflow10)
	begin
	 rx<=32'b00100010000000000000000000000001;
	end
	else begin
	 rx<=r;end

	if(overflow11||underflow11)
	begin
	 sx<=32'b00100010000000000000000000000001;
	end
	else begin
	 sx<=s;end

	if(overflow12||underflow12)
	begin
	 tx<=32'b00100010000000000000000000000001;
	end
	else begin
	 tx<=t;end

	if(overflow13||underflow13)
	begin
	 zx<=32'b00100010000000000000000000000001;
	end
	else begin
	 zx<=z;end

	if(overflow14||underflow14)
	begin
	 aax<=32'b00100010000000000000000000000001;
	end
	else begin
	 aax<=aa;end

	if(overflow15||underflow15)
	begin
	 ccx<=32'b00100010000000000000000000000001;
	end
	else begin
	 ccx<=cc;end

	if(overflow16||underflow16)
	begin
	 jjx<=32'b00100010000000000000000000000001;
	end
	else begin
	 jjx<=jj;end
	
	
	//if((~ee)&(~ff)&(~gg)&(~hh))
		begin
		 //tanhx<=v;
		//assign tanhx=32'b00111111011111111111111111011110;
		end

	/*if((ee)&&(~ff)&(~gg)&(~hh))
		begin
		assign tanhx=g;
		end

	if((ee)&&(ff)&(~gg)&(~hh))
		begin
		assign tanhx=k;
		end

	if((ee)&&(ff)&(gg)&(~hh))
		begin
		assign tanhx=p;
		end
	

	if((ee)&&(ff)&(gg)&(hh))
		begin
		assign tanhx=32'b00111111100000000000000000000000;
		end*/

  tanhx<=v;
	S<=dd;
	 
	 
	 //S<=dd;

 end
end
endmodule

// Module for C[n]
module C(
input clk, reset, enable,clkm,
input [31:0] K0,K1,K2,K3,K4,K5,K6,K7,K8,
input [31:0] S,
input [31:0]m,
input [31:0]Wpost,
output reg [31:0] C
);
wire underflow1,overflow1,underflow2,overflow2,underflow3,overflow3,underflow4,overflow4,underflow5,overflow5,underflow6,overflow6,underflow7,overflow7,underflow8,overflow8,underflow9,overflow9,underflow10,overflow10,underflow11,overflow11,underflow12,overflow12,underflow13,overflow13;
wire [31:0] a,b,c,d,e,f,g,h,i,j,k,l,n,o,q,r,s,t,u,v,w,z,y;
reg [31:0] u1,ax,bx,cx,dx,ex,fx,gx,hx,ix,jx,qx,wx,zx,temp_k,actual_k,temp_y,actual_y;
reg [31:0] const13;
reg [31:0] const99;
reg [31:0] k2;
reg [31:0] k0;

controlx MMM7(clkm,K2,S,a,underflow1,overflow1);//K2*S//
controlx MMM8(clkm,K1,Wpost,b,underflow2,overflow2);//K1*Wpost//
//controlx MMM9(clkm,C,C,c,underflow3,overflow3);//C2
//controlx MMM11(clkm,cx,cx,e,underflow4,overflow4);//C4

controlx MMM977867867(clkm,C,const99,z,underflow13,overflow13);//C(0.9)

//controlx MMM14(clkm,m,m,h,underflow8,overflow8);//m2
//controlx MMM12(clkm,hx,m,f,underflow6,overflow6);//m3

//controlx MMM13(clkm,K6,m,g,underflow7,overflow7);//K6*m//
//controlx MMM119(clkm,K7,hx,d,underflow5,overflow5);//K7*m2//
//controlx MMM15(clkm,K8,fx,i,underflow9,overflow9);//K8*m3//
//controlx MMM16(clkm,cx,K3,j,underflow10,overflow10);//K3*C2//
//controlx MMM18(clkm,ex,K4,q,underflow11,overflow11);//K4*C4//

  fpadd AAA6(K0,ax,clkm,k);
//fpadd AAA7(k,bx,clkx,l);
//fpadd AAA8(k,qx,clkx,n);//using k instead of l for Wpost=0
//fpadd AAA9(n,jx,clkx,o);
//fpadd AAA10(o,dx,clkx,r);
//fpadd AAA11(r,gx,clkx,s);
//fpadd AAA13(s,ix,clkx,t);
  fpadd AAA1233(k,zx,clkm,y);//t in place of k
 // fpadd AAA12(C,y,clkm,u);

//controlx MMMNNNNN18(Isyn,const13,w,underflow12,overflow12);

always@(posedge clk)
begin


 if (reset) begin

C<=32'b00110110101001111100010110101100;
ax<=32'b00100010000000000000000000000001;
bx<=32'b00100010000000000000000000000001;
cx<=32'b00100010000000000000000000000001;
dx<=32'b00100010000000000000000000000001;
ex<=32'b00100010000000000000000000000001;
fx<=32'b00100010000000000000000000000001;
gx<=32'b00100010000000000000000000000001;
hx<=32'b00100010000000000000000000000001;
ix<=32'b00100010000000000000000000000001;
jx<=32'b00100010000000000000000000000001;
qx<=32'b00100010000000000000000000000001;
wx<=32'b00100010000000000000000000000001;
zx<=32'b00100010000000000000000000000001;

actual_k<=32'b00100010000000000000000000000001;
actual_y<=32'b00100010000000000000000000000001;

const13<=32'b01010101000100011000010011100111;
const99<=32'b00111111011001100110011001100110;

//k2<=32'b01000001001000000000000000000000;
//k0<=32'b00110011001010111100110001110111;

 end
 
 else if (enable) begin
if(overflow1||underflow1)
begin
 ax<=32'b00100010000000000000000000000001;
end
else begin
 ax<=a;end

if(overflow2||underflow2)
begin
 bx<=32'b00100010000000000000000000000001;
end
else begin
 bx<=b;end

if(overflow3||underflow3)
begin
 cx<=32'b00100010000000000000000000000001;
end
else begin
 cx<=c;end

if(overflow4||underflow4)
begin
 dx<=32'b00100010000000000000000000000001;
end
else begin
 dx<=d;end

if(overflow5||underflow5)
begin
 ex<=32'b00100010000000000000000000000001;
end
else begin
 ex<=e;end

if(overflow6||underflow6)
begin
 fx<=32'b00100010000000000000000000000001;
end
else begin
 fx<=f;end

if(overflow7||underflow7)
begin
 gx<=32'b00100010000000000000000000000001;
end
else begin
 gx<=g;end

if(overflow8||underflow8)
begin
 hx<=32'b00100010000000000000000000000001;
end
else begin
 hx<=h;end

if(overflow9||underflow9)
begin
 ix<=32'b00100010000000000000000000000001;
end
else begin
 ix<=i;end

if(overflow10||underflow10)
begin
 jx<=32'b00100010000000000000000000000001;
end
else begin
 jx<=j;end

if(overflow11||underflow11)
begin
 qx<=32'b00100010000000000000000000000001;
end
else begin
 qx<=q;end

if(overflow12||underflow12)
begin
 wx<=32'b00100010000000000000000000000001;
end
else begin
 wx<=w;end

if(overflow13||underflow13)
begin
 zx<=32'b00100010000000000000000000000001;
end
else 
  begin
    zx<=z;
 end

     
     C<=y;
 end
end
endmodule

//Module for m[n]
module MXX(
	input clk, reset, enable,clkm,
	input [31:0] K3,K4,K5,K6,K7,K8,
	input [31:0] C,
	output reg [31:0] m
);
wire underflow1,overflow1,underflow2,overflow2,underflow3,overflow3,underflow4,overflow4,underflow5,overflow5,underflow6,overflow6,underflow7,overflow7,underflow8,overflow8,underflow9,overflow9;
wire [31:0] a,b,c,d,e,f,g,h,i,j,k,l,p,n,o;
reg [31:0] n1,ax,bx,cx,dx,ex,fx,gx,hx,ix;
controlx MMMX1(clkm,C,C,a,underflow1,overflow1);//C2
controlx MMMX2(clkm,ax,ax,b,underflow2,overflow2);//C4

controlx MMMX4(clkm,m,m,d,underflow4,overflow4);//m2
controlx MMMX3(clkm,dx,m,c,underflow3,overflow3);//m3

controlx MMMX5(clkm,K3,ax,e,underflow5,overflow5);//K3*C2//
controlx MMMX6(clkm,K4,bx,f,underflow6,overflow6);//K4*C4//
controlx MMMX7(clkm,K6,m,g,underflow7,overflow7);//K6*m//
controlx MMMX8(clkm,K7,dx,h,underflow8,overflow8);//K7*m2//
controlx MMMX9(clkm,K8,cx,i,underflow9,overflow9);//K8*m3//

fpadd AAAX1(m,ex,clkm,j);
fpadd AAAX2(j,gx,clkm,k);
fpadd AAAX3(k,hx,clkm,l);
fpadd AAAX4(l,fx,clkm,p);
fpadd AAAX5(p,ix,clkm,n);



always@(posedge clk)
begin

	

 if (reset) 
 begin
   
	m<= 32'b00100010000000000000000000000001;
	ax<=32'b00100010000000000000000000000001;
	bx<=32'b00100010000000000000000000000001;
	cx<=32'b00100010000000000000000000000001;
	dx<=32'b00100010000000000000000000000001;
	ex<=32'b00100010000000000000000000000001;
	fx<=32'b00100010000000000000000000000001;
	gx<=32'b00100010000000000000000000000001;
	hx<=32'b00100010000000000000000000000001;
	ix<=32'b00100010000000000000000000000001;
	
 end 
 
 else if (enable) 
  begin
  	if(overflow1||underflow1)
	begin
	 ax<=32'b00100010000000000000000000000001;
	end
	else begin
	 ax<=a;end
	
	if(overflow2||underflow2)
	begin
	 bx<=32'b00100010000000000000000000000001;
	end
	else begin
	 bx<=b;end
	
	if(overflow3||underflow3)
	begin
	 cx<=32'b00100010000000000000000000000001;
	end
	else begin
	 cx<=c;end

	if(overflow4||underflow4)
	begin
	 dx<=32'b00100010000000000000000000000001;
	end
	else begin
	 dx<=d;end
	
	if(overflow5||underflow5)
	begin
	 ex<=32'b00100010000000000000000000000001;
	end
	else begin
	 ex<=e;end
	
	if(overflow6||underflow6)
	begin
	 fx<=32'b00100010000000000000000000000001;
	end
	else begin
	 fx<=f;end

	if(overflow7||underflow7)
	begin
	 gx<=32'b00100010000000000000000000000001;
	end
	else begin
	 gx<=g;end

	if(overflow8||underflow8)
	begin
	 hx<=32'b00100010000000000000000000000001;
	end
	else begin
	 hx<=h;end
	
	if(overflow9||underflow9)
	begin
	 ix<=32'b00100010000000000000000000000001;
	end
	else begin
	 ix<=i;end

 	 m<=n;

 end
end
endmodule

// Module for Vpost[n]
module Vpost(
	input clknew,clk, reset, enable,clkm,
	input [31:0] C0,
	input [31:0] C1,
	input [31:0] Wpost,
	input [31:0] C,
	output reg [31:0] Vpost
);
wire underflow1,overflow1,underflow2,overflow2,underflow3,overflow3,underflow4,overflow4,underflow5,overflow5,underflow6,overflow6,underflow7,overflow7;
wire [31:0] out,a,b,c,d,e,f,g,h,k,l,m,n;
reg [31:0] const1;
reg [31:0] g1,ax,bx,cx,dx,fx,kx,nx;
reg [31:0] Vx,Vx1,Vx2,Vx3;

reg [31:0] constm1by3del,constmdel,conste4,const0_0014,const1plusdel,constm1by3,constdel,constnew;

//controlx MMM19(clkm,C0,Wpost,a,underflow1,overflow1);//C0*Wpost
//controlx MMM20(clkm,C1,Vpost,b,underflow2,overflow2);
//controlx MMM21(clkm,bx,Vpost,c,underflow3,overflow3);
//controlx MMM22(clkm,cx,Vpost,d,underflow4,overflow4);//C1*Vpost*Vpost*Vpost

//fpadd AAA14(const1,C0,clkm,e);
//controlx MMM23(clkm,e,Vpost,f,underflow5,overflow5);//(C0+1)Vpost

//fpadd AAA15(dx,ax,clkm,g);
//fpadd SSS2(fx,g1,clkm,h);

//square CUBIC(clkm,reset,enable,Vpost,out);
controlx MMM1129(clkm,constnew,Vx,n,underflow7,overflow7);
controlx MMM19(clkm,constm1by3,Vx1,a,underflow1,overflow1);
fpadd SSS2(ax,const1,clkm,m);
controlx MMM20(clkm,m,Vx1,b,underflow2,overflow2);
controlx MMM2293745(clkm,C,conste4,k,underflow6,overflow6);

fpadd AAA14(const0_0014,kx,clkm,e);
fpadd AAA15(e,bx,clkm,g);

controlx MMM21(clkm,g,constdel,c,underflow3,overflow3);
//controlx MMM22(clkm,Wpost,constmdel,d,underflow4,overflow4);
fpadd SSS23123(cx,Vx,clkm,h);

//controlx MMM23(clkm,const1plusdel,Vpost,f,underflow5,overflow5);



//fpadd SSS2(g,dx,clkm,l);


always@(posedge clkm)
begin
 //  g1<={~Wpost[31],Wpost[30:0]};
end


always@(posedge clk)
begin 

 if (reset) begin
	
	Vpost<= 32'b00100010000000000000000000000001;
	ax<=32'b00100010000000000000000000000001;
	bx<=32'b00100010000000000000000000000001;
	cx<=32'b00100010000000000000000000000001;
	dx<=32'b00100010000000000000000000000001;
	fx<=32'b00100010000000000000000000000001;
	kx<=32'b00100010000000000000000000000001;
	nx<=32'b00100010000000000000000000000001;
	const1<=32'b00111111100000000000000000000000;
	constm1by3del<=32'b10111011010110100111001101111110;
	constmdel<=32'b10111100001000111101011100001010;
	conste4<=32'b01000110000111000100000000000000;
	const0_0014<=32'b00111110000011110101110000101000;
	const1plusdel<=32'b00111111100000010100011110101110;
	constm1by3<=32'b10111111000000000000000000000000;
	//const1<=32'b00111111100000000000000000000000;
	constnew<=32'b00111100001000111101011100001010;
	constdel<=32'b00111100001000111101011100001010;
	Vx<=32'b00100010000000000000000000000001;
	Vx1<=32'b00100010000000000000000000000001;
	Vx2<=32'b00100010000000000000000000000001;
	Vx3<=32'b00100010000000000000000000000001;
	
 end 
 else if (enable) begin
	if(overflow1||underflow1)
	begin
	 ax<=32'b00100010000000000000000000000001;
	end
	else begin
	 ax<=a;end
	
	if(overflow2||underflow2)
	begin
	 bx<=32'b00100010000000000000000000000001;
	end
	else begin
	 bx<=b;end
	
	if(overflow3||underflow3)
	begin
	 cx<=32'b00100010000000000000000000000001;
	end
	else begin
	 cx<=c;end

	if(overflow4||underflow4)
	begin
	 dx<=32'b00100010000000000000000000000001;
	end
	else begin
	 dx<=d;end
	
	if(overflow5||underflow5)
	begin
	 fx<=32'b00100010000000000000000000000001;
	end
	else begin
	 fx<=f;end
	 
	 if(overflow6||underflow6)
	begin
	 kx<=32'b00100010000000000000000000000001;
	end
	else begin
	 kx<=k;end 
	 
	 if(overflow7||underflow7)
	begin
	 Vx1<=32'b00100010000000000000000000000001;
	end
	else begin
	 Vx1<=n;end 
	 
	 Vx<=Vpost;
	 Vpost <= h;
	 
 end
end

  
endmodule

// Module for Wpost[n]
module Wpost(
	input clk, reset, enable,clkm,
	input [31:0] C0,
	input [31:0] Vpost,
	input [31:0] Ipost,
	input [31:0] Isyn,
	output reg [31:0] Wpost
);
wire underflow1,overflow1,underflow2,overflow2;
wire [31:0] a,b,c,d,e;
reg [31:0] b1,cx,ex;
reg [31:0] const1m0_00064,const0_0008,const0_00056;

//fpadd AAA16(Vpost,Ipost,clkm,a);
//fpadd SSS3(a,b1,clkm,b);
//controlx MMM24(clkm,C0,b,c,underflow1,overflow1);
//fpadd AAA17(Wpost,c,clkm,d);
controlx MMM24(clkm,Wpost,const1m0_00064,c,underflow1,overflow1);
controlx MMM24123123(clkm,Vpost,const0_0008,e,underflow2,overflow2);

fpadd AAA16(cx,ex,clkm,a);
fpadd SSS3(a,const0_00056,clkm,d);

always@(posedge clkm)
begin
  b1<={~Isyn[31],Isyn[30:0]};
end


always@(posedge clk)
begin

 if (reset) begin
	
	Wpost<=32'b00100010000000000000000000000001;
	cx<=32'b00100010000000000000000000000001;
	ex<=32'b00100010000000000000000000000001;
	const1m0_00064<=32'b00111111011111111101011000001110;
	const0_0008<=32'b00111010010100011011011100010111;
	const0_00056<=32'b00111010000100101100110011110110;
	
 end 
 else if (enable) begin
   
	if(overflow1||underflow1)
	begin
	 cx<=32'b00100010000000000000000000000001;
	end
	else begin
	 cx<=c;end
	 
	 if(overflow2||underflow2)
	begin
	 ex<=32'b00100010000000000000000000000001;
	end
	else begin
	 ex<=e;end
	 
	 Wpost<=d;
 end
end
endmodule

// Module for RM[n]
module RM(
	input clkb, reset, enable,clkm,
	input [31:0] L0,
	input [31:0] L1,
	input [31:0] C,
	output reg [31:0] RM
);
wire underflow1,overflow1,underflow2,overflow2,underflow3,overflow3,underflow4,overflow4,underflow5,overflow5,underflow6,overflow6,underflow7,overflow7,underflow8,overflow8,underflow9,overflow9,underflow10,overflow10;
wire [31:0] a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p;
reg [31:0]ax,bx,cx,dx,ex,fx,gx,lx,mx,nx,g1;
reg [31:0] conste12;
reg [31:0] conste1by2;
reg [31:0] conste1by6m;
reg [31:0] constm1;
reg [31:0] const1;
reg [31:0] const02e6;
reg [31:0] constm5e6;
reg [31:0] constm4e12;
reg [31:0] const8e18;
integer count;

//reg [31:0] C;


/*
fpadd ANNA4(C,constm5e6,clkm,k);
controlx MNNM8(clkm,k,k,l,underflow8,overflow8);
controlx MNNM9(clkm,k,constm4e12,m,underflow9,overflow9);
controlx MNNM10(clkm,lx,const8e18,n,underflow10,overflow10);
fpadd ANNA5(mx,const02e6,clkm,o);
*/

Division DUT123(clkm,const1,C,Exception,p);

//fpadd ANNA6(o,nx,clkm,p);


controlx MNNM1(clkm,p,p,a,underflow1,overflow1);//1/(C*C)
controlx MNNM2(clkm,ax,conste12,b,underflow2,overflow2);//x
controlx MNNM3(clkm,bx,bx,c,underflow3,overflow3);//x*x
controlx MNNM4(clkm,cx,bx,d,underflow4,overflow4);//x*x*x
controlx MNNM5(clkm,cx,conste1by2,e,underflow5,overflow5);//x*x/2
controlx MNNM6(clkm,dx,conste1by6m,f,underflow6,overflow6);//-x*x*x/6
controlx MNNM7(clkm,bx,constm1,g,underflow7,overflow7);//-x
fpadd ANNA1(const1,gx,clkm,h);
fpadd ANNA2(h,fx,clkm,i);
fpadd ANNA3(i,ex,clkm,j);

always@(posedge clkm)
begin
  
   g1<={~ax[31],ax[30:0]};
  
  
end



always@(posedge clkb)
begin


 if (reset) begin
	//C<=32'b00110110101001111100010110101100;
	RM<=32'b00100010000000000000000000000001;
	ax<=32'b00100010000000000000000000000001;
	bx<=32'b00100010000000000000000000000001;
	cx<=32'b00100010000000000000000000000001;
	dx<=32'b00100010000000000000000000000001;
	ex<=32'b00100010000000000000000000000001;
	fx<=32'b00100010000000000000000000000001;
	gx<=32'b00100010000000000000000000000001;
	lx<=32'b00100010000000000000000000000001;
	mx<=32'b00100010000000000000000000000001;
	nx<=32'b00100010000000000000000000000001;
	g1<=32'b00100010000000000000000000000001;
	
 conste12<=32'b00101101001000010111111010011000;
 conste1by2<=32'b00111111000000000000000000000000;
 conste1by6m<=32'b10111110001010101010101010101010;
 constm1<=32'b10111111100000000000000000000000;
 const1<=32'b00111111100000000000000000000000;
 const02e6<=32'b01001000010000110101000000000000;
 constm5e6<=32'b10110110101001111100010110101100;
 constm4e12<=32'b11010001000101010000001011111001;
 const8e18<=32'b01011001111000110101111110101001;
 
 count<=0;

 end 
 else if (enable) begin
	if(overflow1||underflow1)
	begin
	 ax<=32'b00100010000000000000000000000001;
	end
	else begin
	 ax<=a;end

	if(overflow2||underflow2)
	begin
	 bx<=32'b00100010000000000000000000000001;
	end
	else begin
	 bx<=b;end

	if(overflow3||underflow3)
	begin
	 cx<=32'b00100010000000000000000000000001;
	end
	else begin
	 cx<=c;end

	if(overflow4||underflow4)
	begin
	 dx<=32'b00100010000000000000000000000001;
	end
	else begin
	 dx<=d;end

	if(overflow5||underflow5)
	begin
	 ex<=32'b00100010000000000000000000000001;
	end
	else begin
	 ex<=e;end
	
	if(overflow6||underflow6)
	begin
	 fx<=32'b00100010000000000000000000000001;
	end
	else begin
	 fx<=f;end

	if(overflow7||underflow7)
	begin
	 gx<=32'b00100010000000000000000000000001;
	end
	else begin
	 gx<=g;end

	if(overflow8||underflow8)
	begin
	 lx<=32'b00100010000000000000000000000001;
	end
	else begin
	 lx<=l;end
	
	if(overflow9||underflow9)
	begin
	 mx<=32'b00100010000000000000000000000001;
	end
	else begin
	 mx<=m;end

	if(overflow10||underflow10)
	begin
	 nx<=32'b00100010000000000000000000000001;
	end
	else begin
	 nx<=n;end

	if(count<=25)
	  begin
	  count<=count+1;
	  RM<=32'b00100010000000000000000000000001;
	  end
	  
  else
    begin
	   RM<=j;
	   end
	   
 end //elseif
end //always
endmodule



module PostSynapse( input clknew,tp,clk1, reset1, enable1,clkx1,
	input [31:0] Isyn,
	
	input clk2, reset2, enable2,clkx2,
	
	
	input clk3, reset3, enable3,clkx3,
	input [31:0] Isyn3,
	
	input clk4, reset4, enable4,clkx4,

	
	input clkb, reset5, enable5,clkx5,
	output  [31:0] RM5,
	
	input clk6, reset6, enable6,clkx6,
	output [31:0]Cx,Sx,Mx,Vpostx,Wpostx,tanhxs,Isyns,xs,ws,ys,zs,ps,qs,rs
);



wire [31:0] CtoS,Vpost,Wpost,Cout,CtoM,MtoC;

reg[31:0] K01;
reg[31:0]K11;
reg[31:0]K21;
reg[31:0]K31;
reg[31:0]K41;

reg[31:0]C02;
reg[31:0]C12;

reg [31:0]C03;
reg[31:0]Ipost3;

reg [31:0]K04;//
reg [31:0]K14;//
reg [31:0]K24;//
reg [31:0]K34;//
reg [31:0]K44;//
reg [31:0]K54;
//reg [31:0]K54=32'b00001001001110000111011000111000;//
reg [31:0]K64;//
reg [31:0]K74;//
reg [31:0]K84;//

reg [31:0]L15;
reg [31:0]L05;
reg [31:0]K36;//
reg [31:0]K46;//
reg [31:0]K56;
//reg [31:0]K56=32'b00001001001110000111011000111000;//
reg [31:0]K66;//
reg [31:0]K76;//
reg [31:0]K86;//

assign Cx=CtoM;
assign Sx=CtoS;
assign Mx=MtoC;
assign Vpostx=Vpost;
assign Wpostx=Wpost;
assign RMx=RM5;



S S1(clk1,reset1,enable1,clkx1,Isyn,tp,CtoS,tanhxs,Isyns,xs,ws,ys,zs,ps,qs,rs);
Vpost V1(clknew,clk2,reset2,enable2,clkx2,C02,C12,Wpost,CtoM,Vpost);

Wpost Wp(clk3,reset3,enable3,clkx3,C03,Vpost,Ipost3,Isyn3,Wpost);
MXX  X1(clk6, reset6, enable6,clkx6, K36,K46,K56,K66,K76,K86,CtoM, MtoC);
C C1(clk4,reset4,enable4,clkx4,K04,K14,K24,K34,K44,K54,K64,K74,K84,CtoS,MtoC,Wpost,CtoM);

RM RM1(clkb ,reset5, enable5,clkx5,L05, L15,CtoM,RM5);

always@(posedge clk1)
begin
  if(reset1)begin
  
  K01<=32'b11000010000000010101010101001100;
  K11<=32'b00110011000011110010011010111000;
  K21<=32'b00000000100000000000000000000000;
  K31<=32'b01000100011110100000000000000000;
  K41<=32'b00111101000010001000100000101111;

  C02<=32'b00101111110110111110011011111110;//4e-9        //(4e-10)
  C12<=32'b00101111000100101001100111111100;//1.333e-9    //(1.333e-10)

  C03<=32'b00110000100010010111000001011111;//1e-8      //(1e-9)
  Ipost3<=32'b00111111100001100110011001100110;

  K04<=32'b00110010001010111100110001110111;//4e-7      
  K14<=32'b00110001010011100010100010001110;//3e-8      //(3e-9)
  K24<=32'b01000001001000000000000000000000;//100        
  K34<=32'b01000111100100111001000111000111;//7.55e5      //(7.55e4)
  K44<=32'b11010111100110001011010001000011;//-3.358e15       //(-3.358e14)
  K54<=32'b10111101111101011100001010001111;//-1.2            //(-1.2e-1)
  //reg [31:0]K54=32'b00001001001110000111011000111000;//
  K64<=32'b10111101111101011100001010001111;//-1.2            //(-1.2e-1)
  K74<=32'b01000111111010100110000000000000;//1.2e6       //(1.2e5)
  K84<=32'b11010001110111111000010001110101;//-1.2e12     //(-1.2e11)
  
  L15<=32'b00111111100000000000000000000000;
  L05<=32'b00101101011011101001100100110100;
  
  K36<=32'b01000111100100111001000111000111;//7.55e5      //(7.55e4)
  K46<=32'b11010111100110001011010001000011;//-3.358e15       //(-3.358e14)
  K56<=32'b10111101111101011100001010001111;//-1.2            //(-1.2e-1)
  //reg [31:0]K54=32'b00001001001110000111011000111000;//
  K66<=32'b10111101111101011100001010001111;//-1.2            //(-1.2e-1)
  K76<=32'b01000111111010100110000000000000;//1.2e6       //(1.2e5)
  K86<=32'b11010001110111111000010001110101;//-1.2e12     //(-1.2e11)
  
  end
end

endmodule


module MODEL(
	input clknew,clkm,clks,clkp,clkxp,clka,clkb,clkxy,clk, reset, enable,clkx,
	input tp1,	td4,
	output[31:0] Isyn1,RMx,WWx,Ax,Dx,RMtrx,Prelx,Inhx,Pxy,Xn,Yn,Cx,Sx,Mx,Vpostx1,Wpostx1,xs,tanhxs,ws,ys,zs,ps,qs,rs,RMout);

wire [31:0] Isyns;
reg [31:0] RM;

//assign RMx=RM;

PreSynapse PS1(clkm,clkxy,clkb, reset, enable,clkx, tp1,clkb, reset, enable, clkx,clkb, reset, enable,clkx, RMx, clka, reset, enable,clkx,td4,clkp, reset, enable,clkxp,clk, reset, enable,clkx,clk, reset, enable,clkx, clks, reset, enable,clkx,tp1,Isyn1,WWx,Ax,Dx,RMtrx,Prelx,Inhx,Pxy,Xn,Yn);
PostSynapse PS2(clknew,tp1,clkb, reset, enable,clkm, Isyn1, clkb, reset, enable, clkm, clkb, reset, enable,clkm,Isyn1, clkb, reset, enable,clkm,clkb, reset, enable,clkm, RMx, clkb, reset, enable,clkm,Cx,Sx,Mx,Vpostx1,Wpostx1,tanhxs,Isyns,xs,ws,ys,zs,ps,qs,rs);



endmodule


//(Please add the clock clm=0.1ps and make necessary changes)




//Top level design module with all instances
module top(
clknew,clk_0_1ps,reset,enable,td4,tp1,Isyn1,RMx,WWx,Ax,Dx,RMtrx,Prelx,Inhx,Pxy,Xn,Yn,Cx,Sx,Mx,Vpostx1,Wpostx1,xs,tanhxs,ws,ys,zs,ps,qs,rs,RMout
);
  //Input declaration
  input clknew,clk_0_1ps;
  input reset;
  input enable;
  input td4;
  input tp1;
  //Output declarations
  output [31:0]       Isyn1,RMx,WWx,Ax,Dx,RMtrx,Prelx,Inhx,Pxy,Xn,Yn,Cx,Sx,Mx,Vpostx1,Wpostx1,xs,tanhxs,ws,ys,zs,ps,qs,rs,RMout;
  //Reg variables
  //reg clk,reset,enable,clkx,td4,tp1,clkxy,clka,clkb,clkp,clkxp,clks; //inputs 
  //reg clk_1ps;
  reg clk,clkxy,clka,clkb,clkp,clkxp,clks; //inputs 
  //wire variables
 wire [31:0] Isyn1,RMx,WWx,Ax,Dx,RMtrx,KKx,Prelx,Inhx,Pxy,X,Y,Cx,Sx,Mx,Vpostx1,Wpostx1,xs,tanhxs,ws,ys,zs,ps,qs,rs,RMout;
  parameter THRESHOLD_FOR_5NS_CLOCK = 500000;//Always should be even//(0.1ps=0.0001ns)*50000=5ns
  parameter THRESHOLD_FOR_2_5NS_CLOCK = 250000;//Always should be even//(0.1ps=0.0001ns)*25000=2.5ns
  parameter THRESHOLD_FOR_0_1NS_CLOCK = 10000;//Always should be even//(0.1ps=0.0001ns)*1000=0.1ns
  //parameter THRESHOLD_FOR_0_1PS_CLOCK = 100;//Always should be even//(1fs=0.0001ns)*100=(0.0001ns=0.1ps)
  MODEL SynNeur1(.clknew(clknew),.clkm(clk_0_1ps),.clks(clks),.clkp(clks),.clkxp(clkxp),.clka(clks),.clkb(clks),.clkxy(clkxp),.clk(clk),.reset(reset), .enable(enable),.clkx(clkxp), .tp1(tp1),.td4(td4), .Isyn1(Isyn1),.RMx(RMx),.WWx(WWx),.Ax(Ax),.Dx(Dx),.RMtrx(RMtrx),.Prelx(Prelx),.Inhx(Inhx),.Pxy(Pxy),.Xn(Xn),.Yn(Yn),.Cx(Cx),.Sx(Sx),.Mx(Mx),.Vpostx1(Vpostx1),.Wpostx1(Wpostx1),.xs(xs),.tanhxs(tanhxs),.ws(ws),.ys(ys),.zs(zs),.ps(ps),.qs(qs),.rs(rs),.RMout(RMout)); //instantiate
  ///////////////////////Clocks coming from top level clkgen module./////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////
  clk_gen #(.THRESHOLD_FOR_CLOCK(THRESHOLD_FOR_5NS_CLOCK))
  gen_instance5 (.clk_0_1ps(clk_0_1ps),.clk_out(clk));
  
 // clk_gen #(.THRESHOLD_FOR_CLOCK(THRESHOLD_FOR_2_5NS_CLOCK))
 // gen_instance2_5a (.clk_0_1ps(clk_0_1ps),.clk_out(clka));
 // 
 // clk_gen #(.THRESHOLD_FOR_CLOCK(THRESHOLD_FOR_2_5NS_CLOCK))
 // gen_instance2_5b (.clk_0_1ps(clk_0_1ps),.clk_out(clkb));
 // 
 // clk_gen #(.THRESHOLD_FOR_CLOCK(THRESHOLD_FOR_2_5NS_CLOCK))
 // gen_instance2_5p (.clk_0_1ps(clk_0_1ps),.clk_out(clkp));
  
  
  clk_gen #(.THRESHOLD_FOR_CLOCK(THRESHOLD_FOR_2_5NS_CLOCK))
  gen_instance2_5s (.clk_0_1ps(clk_0_1ps),.clk_out(clks));
  
 // clk_gen #(.THRESHOLD_FOR_CLOCK(THRESHOLD_FOR_0_1NS_CLOCK))
 // gen_instance0_1xy (.clk_0_1ps(clk_0_1ps),.clk_out(clkxy));
 // 
 // clk_gen #(.THRESHOLD_FOR_CLOCK(THRESHOLD_FOR_0_1NS_CLOCK))
 // gen_instance0_1x (.clk_0_1ps(clk_0_1ps),.clk_out(clkx));
  
  clk_gen #(.THRESHOLD_FOR_CLOCK(THRESHOLD_FOR_0_1NS_CLOCK))
  gen_instance0_1xp (.clk_0_1ps(clk_0_1ps),.clk_out(clkxp));

  //clk_gen #(.THRESHOLD_FOR_CLOCK(THRESHOLD_FOR_0_1PS_CLOCK))
  //gen_instance0_1m (.clk_1fs(clk_1fs),.clk_out(clkm));
endmodule
