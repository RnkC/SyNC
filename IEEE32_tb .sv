 `timescale 1ps / 1fs


 module testmodel();
 //reg clk,reset,enable,clkx,td4,tp1,clkxy,clka,clkb,clkp,clkxp,clks; //inputs 
 reg clknew,clk_0_1ps,reset,enable,tp1,td4;
 wire [31:0] Isyn1,RMx,WWx,Ax,Dx,RMtrx,KKx,Prelx,Inhx,Pxy,Xn,Yn,Cx,Sx,Mx,Vpostx1,Wpostx1,xs,tanhxs,ws,ys,zs,ps,qs,rs,RMout;
 integer f,i,j;
   
   top top_inst(.clknew(clknew),.clk_0_1ps(clk_0_1ps),.reset(reset),.enable(enable),.tp1(tp1),.td4(td4),.Isyn1(Isyn1),.RMx(RMx),.WWx(WWx),.Ax(Ax),.Dx(Dx),.RMtrx(RMtrx),.Prelx(Prelx),.Inhx(Inhx),.Pxy(Pxy),.Xn(Xn),.Yn(Yn),.Cx(Cx),.Sx(Sx),.Mx(Mx),.Vpostx1(Vpostx1),.Wpostx1(Wpostx1),.xs(xs),.tanhxs(tanhxs),.ws(ws),.ys(ys),.zs(zs),.ps(ps),.qs(qs),.rs(rs),.RMout(RMout));
 //parameter THRESHOLD_FOR_5NS_CLOCK = 5000;//Always should be even//0.001*5000=5ns
 //parameter THRESHOLD_FOR_2_5NS_CLOCK = 2500;//Always should be even//0.001*2500=2.5ns
 //parameter THRESHOLD_FOR_0_1NS_CLOCK = 100;//Always should be even//0.001*100=0.1ns

//MODEL SynNeur1(.clks(clks),.clkp(clkp),.clkxp(clkxp),.clka(clka),.clkb(clkb),.clkxy(clkxy),.clk(clk),.reset(reset), .enable(enable),.clkx(clkx), .tp1(tp1),.td4(td4), .Isyn1(Isyn1),.RMx(RMx),.Cx(Cx),.WWx(WWx),.Ax(Ax),.Dx(Dx),.RMtrx(RMtrx),.KKx(KKx),.Prelx(Prelx),.Inhx(Inhx),.Sx(Sx),.Mx(Mx),.Vpostx1(Vpostx1),.Wpostx1(Wpostx1),.aq(aq),.bq(bq),.cq(cq),.dq(dq),.eq(eq),.fq(fq),.gq(gq),.hq(hq),.iq(iq),.Sq(Sq),.S1q(S1q),.S11q(S11q),.k0q(k0q),.ax(ax),.bx(bx),.cx(cx),.dx(dx),.ex(ex),.fx(fx),.gx(gx),.anq(anq),.bnq(bnq),.cnq(cnq),.dnq(dnq),.enq(enq),.inq(inq),.jnq(jnq),.knq(knq),.mnq(mnq),.nnq(nnq),.n1nq(n1nq),.pnq(pnq),.qnq(qnq),.unq(unq),.u1nq(u1nq),.vnq(vnq),.Sk(Sk),.ak(ak),.counter1(counter1),.gk(gk),.Pxy(Pxy),.kk(kk),.qk(qk),.xk(xk)); //instantiate
  
///////////////////////Clocks coming from top level clkgen module./////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////
  //clk_gen #(.THRESHOLD_FOR_CLOCK(THRESHOLD_FOR_5NS_CLOCK))
  //gen_instance5 (.clk_1ps(clk_1ps),.clk_out(clk));
  
  //clk_gen #(.THRESHOLD_FOR_CLOCK(THRESHOLD_FOR_2_5NS_CLOCK))
  //gen_instance2_5a (.clk_1ps(clk_1ps),.clk_out(clka));
  
  //clk_gen #(.THRESHOLD_FOR_CLOCK(THRESHOLD_FOR_2_5NS_CLOCK))
  //gen_instance2_5b (.clk_1ps(clk_1ps),.clk_out(clkb));
  
  //clk_gen #(.THRESHOLD_FOR_CLOCK(THRESHOLD_FOR_2_5NS_CLOCK))
  //gen_instance2_5p (.clk_1ps(clk_1ps),.clk_out(clkp));
  
  
  //clk_gen #(.THRESHOLD_FOR_CLOCK(THRESHOLD_FOR_2_5NS_CLOCK))
  //gen_instance2_5s (.clk_1ps(clk_1ps),.clk_out(clks));
  
  //clk_gen #(.THRESHOLD_FOR_CLOCK(THRESHOLD_FOR_0_1NS_CLOCK))
  //gen_instance0_1xy (.clk_1ps(clk_1ps),.clk_out(clkxy));
  
  //clk_gen #(.THRESHOLD_FOR_CLOCK(THRESHOLD_FOR_0_1NS_CLOCK))
  //gen_instance0_1x (.clk_1ps(clk_1ps),.clk_out(clkx));
  
  //clk_gen #(.THRESHOLD_FOR_CLOCK(THRESHOLD_FOR_0_1NS_CLOCK))
  //gen_instance0_1xp (.clk_1ps(clk_1ps),.clk_out(clkxp));
  
  ///////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////
  
//always #5ns clk_20ns = ~clk_20ns;//Global clock
  
always #0.01ps clk_0_1ps = ~clk_0_1ps;//Global clock
always #5ns clknew = ~clknew;
  //1 ps clock - reference lowest clock out of which all others are derived
  //////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////
/* always
begin
clkxy = 1;
forever #0.1ns clkxy = ~clkxy;
  end
  
  always
begin
clk = 1;
forever #5ns clk = ~clk;
  end
  
  always
begin
clka = 1;
forever #2.5ns clka = ~clka;
  end
  
   always
begin
clkb = 1;
forever #2.5ns clkb = ~clkb;
  end

always
begin
clkx = 1;
forever #0.1ns clkx = ~clkx;
  end
  
  always
begin
clkxp = 1;
forever #0.1ns clkxp = ~clkxp;
  end
  
  always
begin
clkp = 1;
forever #2.5ns clkp = ~clkp;
  end
  
   always
begin
clks = 1;
forever #2.5ns clks = ~clks;
  end*/
  
   always @(posedge top_inst.clk) begin
    if(enable)begin
     	
     	//$fwrite(outfile, "%h\t%h\t%h\t%h\t%d\n",in1, in2, out,result[i-1],diff);
      $strobe("%b\t%b\t",Vpostx1,top_inst.SynNeur1.PS2.V1.Vx1,$realtime);
     	end
  end
  
  
initial begin
  //$monitor("%b\t",top_inst.SynNeur1.PS2.V1.clknew,$realtime);
  //$monitor("%b\t%b\t%b\t%b\t%b\t%b\t%b\t%b\t%b\t%b\t",Isyn1,Sx,xs,tanhxs,ps,qs,ws,ys,zs,rs,$realtime);
  //integer s;
  //$dumpfile("dump.vcd");
  //$dumpvars(0);
//$monitor("%b\t%b\t",Ax,Dx,$realtime);
  //#1000000ps;
  //$display("%b\t%b\t%b\t%b\t%b\t%b\t%b\t%b\t%b\t%b\t%b\t%b\t%t",RMx,RMtrx,Inhx,Pxy,Prelx,WWx,Ax,Dx,KKx,Isyn1,Cx,Sx,$realtime);
  //$monitor("%b\t%b\t%b\t%b\t%b\t%b\t%b\t%b\t%b\t%b\t%b\t%b\t%b\t%b\t",RMx,RMtrx,Inhx,Pxy,Prelx,WWx,Ax,Dx,KKx,Isyn1,Cx,Sx,$time);
  //for ( s = 0;s<10000;s=s+1)begin
  //#10000ps;    //$display("%b\t%b\t%b\t%b\t%b\t%b\t%b\t%b\t%b\t%b\t%b\t%b\t%t",RMx,RMtrx,Inhx,Pxy,Prelx,WWx,Ax,Dx,KKx,Isyn1,Cx,Sx,$realtime);
  //end
clk_0_1ps = 1;clknew=1;
td4=0;tp1=0;reset=1;enable=0;
#985ns ;
td4=~td4;tp1=~tp1;

#5ns;
td4=~td4;

#5ns;
tp1=~tp1;

#5ns;
 reset=0;enable=1;



for(j=0;j<10;j=j+1)
begin

tp1=~tp1;

#10ns;
tp1=~tp1;td4=~td4;

#10ns;
td4=~td4;

#980ns;

end
 #3us;
$finish;
 
 end       
    
endmodule