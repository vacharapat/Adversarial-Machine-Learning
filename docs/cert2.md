{% include lib/mathjax.html %}
# Linear programming relaxation

ในหัวข้อนี้เราจะมาดูตัวอย่างการ relax ปัญหาการโจมตีแบบกำหนดเป้าหมายให้อยู่ในรูปของ linear program ซึ่งเป็นปัญหา optimization ที่มี objective function เป็น linear function บนตัวแปรที่ปรับค่าได้ และเงื่อนไขทั้งหมดก็อยู่ในรูป linear inequality บนตัวแปรที่ปรับค่าได้เช่นกัน
ในปัจจุบันเรามีอัลกอริทึมสำหรับแก้ปัญหาที่เป็น linear program ได้อย่างมีประสิทธิภาพ ตัวอย่างอัลกอริทึมดังกล่าวเช่น ellipsoid algorithm หรือ interior-point algorithms

เพื่อความสะดวก เราจะแสดง constrained formulation ของปัญหาที่เราสนใจใหม่ดังนี้

$$
\begin{array}{ll}
\max_{z_1,\dots,z_{d+1}}& (e_{y'}-e_y)^Tz_{d+1}\\
\text{subject to}&\\
&\|z_1-x\|_\infty\leq\epsilon\\
&z_{i+1} = \max(0, W_iz_i+b_i), \text{ for } i=1,\dots, d-1\\
&z_{d+1}=W_dz_d+b_d
\end{array}
$$

## linear integer programming
เราจะเริ่มจากการแปลงปัญหาให้อยู่ในรูปของ linear constrain ทั้งหมดก่อน โดยที่ตัวแปรบางตัวอาจมีค่าที่เป็นไปได้ถูกจำกัดให้เป็นจำนวนเต็ม เราเรียกปัญหาลักษณะนี้ว่า linear integer programming

จาก formulation ของปัญหา จะเห็นว่าเงื่อนไขที่ยังไม่เป็น linear constrain ได้แก่สองเงื่อนไขแรก โดยที่เงื่อนไข $$\|z_1-x\|_\infty\leq\epsilon$$ นั้นสามารถแปลงให้อยู่ในรูปของ linear constrain ได้ไม่ยาก โดยการแบ่งเป็นสองเงื่อนไขใหม่ดังนี้

$$
z_1\leq x+\epsilon\\
z_1\geq x-\epsilon
$$

สำหรับเงื่อนไข $$z_{i+1}=\max(0,W_iz_i+b_i)$$ เราจะแปลงให้เป็นชุดของเงื่อนไขที่เป็น linear constrain และ binary integer constrain โดยสมมติว่าเราทราบว่าค่าของ $$W_iz_i+b_i$$ นั้นจะมีค่าไม่น้อยกว่า $$l_i$$ และไม่มากกว่า $$u_i$$ เราสามารถแทนเงื่อนไข $$z_{i+1}=\max(0,W_iz_i+b_i)$$ ด้วยชุดของเงื่อนไขต่อไปนี้ได้ โดยที่ $$v_i$$ เป็นตัวแปรที่มีค่าเป็น 0 หรือ 1 ได้เท่านั้น

$$
\begin{split}
z_{i+1}&\geq&W_iz_i+b_i\\
z_{i+1}&\geq&0\\
u_i\cdot v_i&\geq&z_{i+1}\\
W_iz_i+b_i&\geq&z_{i+1}+(1-v_i)l_i\\
v_i&\in&\{0,1\}^{|v_i|}
\end{split}
$$

เพื่อทำความเข้าใจชุดเงื่อนไขเหล่านี้ เราจะแยกการวิเคราะห์ออกเป็นกรณีย่อยโดยสมมติว่าตัวแปรเหล่านี้เป็นสเกลาร์ทั้งหมดเพื่อความสะดวก

กรณีแรก พิจารณาเมื่อ $$W_iz_i+b_i> 0$$ สังเกตว่าถ้าเรากำหนดให้ $$v_i=0$$ เงื่อนไขที่สามแสดงว่า $$z_{i+1}\leq 0$$
แต่จากเงื่อนไขแรก $$z_{i+1}\geq W_iz_i+b_i>0$$ ซึ่งขัดแย้งกัน แสดงว่า $$v_i$$ ต้องมีค่าเป็น 1 เท่านั้น
ซึ่งเมื่อเรากำหนด $$v_i=1$$ แล้ว เงื่อนไขทั้งหมดจะเหลือเป็น

$$
\begin{split}
z_{i+1}&\geq&W_iz_i+b_i\\
z_{i+1}&\geq&0\\
u_i&\geq&z_{i+1}\\
W_iz_i+b_i&\geq&z_{i+1}\\
\end{split}
$$

เงื่อนไขแรกและเงื่อนไขสุดท้ายจะทำให้เราได้ว่า $$z_{i+1}=W_iz_i+b_i$$
ซึ่งจะต้องสอดคล้องกับเงื่อนไขที่สองเนื่องจาก $$W_iz_i+b_i>0$$
และเงื่อนไขที่สามก็ต้องเป็นจริงด้วยเนื่องจาก $$u_i$$ เป็นขอบเขตบนของ $$W_iz_i+b_i$$
ดังนั้น $$u_i\geq W_iz_i+b_i=z_{i+1}$$

คราวนี้ลองพิจารณาเมื่อ $$W_iz_i+b_i<0$$ สังเกตว่าหากเรากำหนดให้ $$v_i=1$$
เงื่อนไขที่สี่จะทำให้ $$z_{i+1}\leq W_iz_i+b_i<0$$ ซึ่งขัดแย้งกับเงื่อนไขที่สองที่กำหนดให้
$$z_{i+1}\geq 0$$ ดังนั้นจึงได้ว่า $$v_i$$ ต้องเป็น 0 และเงื่อนไขของเราจัดรูปใหม่ได้ดังนี้

$$
\begin{split}
z_{i+1}&\geq&W_iz_i+b_i\\
z_{i+1}&\geq&0\\
0&\geq&z_{i+1}\\
W_iz_i+b_i&\geq&z_{i+1}+l_i\\
\end{split}
$$

เงื่อนไขที่สองและสามจะบังคับให้ $$z_{i+1}=0$$ ซึ่งจะทำให้เงื่อนไขแรกเป็นจริงเนื่องจาก $$W_iz_i+b_i<0$$
และเงื่อนไขที่สี่ก็เป็นจริงด้วยเพราะ $$l_i$$ เป็นขอบเขตล่างของ $$W_iz_i+b_i$$
นั่นคือ $$l_i\leq W_iz_i+b_i$$ แน่นอน

ถึงตรงนี้ เราสามารถแปลงปัญหาของเราเป็น linear integer programming  ได้ดังนี้

$$
\begin{array}{lll}
\max_{z_1,\dots,z_{d+1}}& (e_{y'}-e_y)^Tz_{d+1}&\\
\text{subject to}&&\\
&z_1\leq x+\epsilon&\\
&z_1\geq x-\epsilon&\\
&z_{i+1}\geq W_iz_i+b_i, &\text{ for } i=1,\dots,d-1\\
&z_{i+1}\geq 0, &\text{ for } i=1,\dots,d-1\\
&u_i\cdot v_i\geq z_{i+1}, &\text{ for } i=1,\dots,d-1\\
&W_iz_i+b_i\geq z_{i+1}+ (1-v_i)l_i, &\text{ for } i=1,\dots,d-1\\
&v_i\in\{0,1\}^{|v_i|}, &\text{ for } i=1,\dots,d-1\\
&z_{d+1}=W_dz_d+b_d
\end{array}
$$

เมื่อเราเขียนปัญหาในรูป linear integer programming เช่นนี้ได้
เราสามารถ relax ปัญหานี้ให้เป็น linear programming ได้ง่ายโดยการเปลี่ยนเงื่อนไขขอบเขตของ $$v_i$$
จากที่ต้องมีค่าเป็น 0 หรือ 1 เท่านั้น ให้สามารถมีค่าเป็นเท่าใดก็ได้ตั้งแต่ 0 ถึง 1 นั่นคือ
เราเปลี่ยนเงื่อนไข $$v_i\in\{0,1\}^{|v_i|}$$ ให้กลายเป็น
$$0\leq v_i\leq 1$$ แทน  ภาพด้านล่างแสดงลักษณะการ relax ฟังก์ชัน ReLU ด้วยวิธีนี้ ซึ่งจะเพิ่มพื้นที่ของตัวแปรขึ้นตามพื้นที่แรเงาในภาพด้านขวา
คราวนี้เราก็จะได้ปัญหา linear programming ที่สามารถหาคำตอบได้เร็ว
โดยที่คำตอบของปัญหาตั้งต้นของเราก็เป็นคำตอบที่เป็นไปได้เช่นกัน ดังนั้นเมื่อแก้ linear program
นี้ออกมาเราจะได้ผลลัพธ์ที่เป็นขอบเขตบนของคำตอบที่เราต้องการ

<p align="center">
<img width="500" src="https://raw.githubusercontent.com/vacharapat/Adversarial-Machine-Learning/master/images/lprelax.png">
</p>

## การหาขอบเขตบนและขอบเขตล่าง

ในการทำ linear programming relaxation ที่กล่าวมานี้เราทำได้โดยสมมติว่าทราบขอบเขตบนและขอบเขตล่างของ $$W_iz_i+b_i$$
ทั้งหมดก่อนแล้ว ซึ่งในความเป็นจริงขอบเขตทั้งหมดนี้สามารถประมาณได้หลายวิธี
วิธีหนึ่งที่ทำได้ง่ายคือ หากเราทราบขอบเขตของ $$z$$ ว่า $$l\leq z\leq u$$
เราสามารถหาขอบเขตที่เป็นไปได้ของ $$Wz+b$$ ได้โดยพิจารณาดังนี้

เนื่องจากสมาชิกตัวที่ $$i$$ ใน $$Wz+b$$ จะมีค่าเท่ากับ

$$
\sum_j W_{ij}z_j+b_i
$$

ดังนั้นจะเห็นว่าหากเราต้องการทำให้ค่าของ $$Wz+b$$ น้อยที่สุดเมื่อ $$z$$ อยู่ในช่วง
$$[l,u]$$ โดย $$u\geq 0$$ เราทำได้โดยสังเกตที่ $$W_{ij}$$ ถ้า $$W_{ij}<0$$
เราเลือกให้ $$z_j$$ มากที่สุดเป็น $$u_j$$ ในขณะที่ถ้า $$W_{ij}>0$$ เราเลือกให้
$$z_j$$ มีค่าน้อยที่สุดเป็น $$l_j$$ ด้วยแนวคิดนี้ เราสามารถสรุปได้ว่า

$$
Wz+b\geq \max(W,0)l+\min(W,0)u+b\\
Wz+b\leq \max(W,0)u+\min(W,0)l+b
$$

เราสามารถใช้แนวคิดนี้ในการคำนวณหาขอบเขตบนและขอบเขตล่างทั้งหมดที่ต้องการได้
โดยเมื่อเรานำ $$z$$ ใด ๆ ที่รู้ขอบเขตแล้ว ผ่านเข้าไปยังฟังก์ชัน $$\text{ReLU}$$ เราก็สามารถ clip
ขอบเขตของ $$\text{ReLU}(z)$$ เพื่อนำไปใช้หาขอบเขตใน layer ถัดไปได้ไม่ยาก

## References
1. [E. Wong, J.Z. Kolter. Provable defenses against adversarial examples via the convex outer adversarial polytope,
In Proceedings of the International Conference on Machine Learning (ICML), 2018](https://arxiv.org/abs/1711.00851)

---
Prev: [การสร้าง robustness certificate](https://vacharapat.github.io/Adversarial-Machine-Learning/docs/cert1)

Next: [Quadratic programming](https://vacharapat.github.io/Adversarial-Machine-Learning/docs/cert3)
