{% include lib/mathjax.html %}
# Semidefinite programming relaxation

เมื่อเราเขียนปัญหาการโจมตีแบบกำหนดเป้าหมายให้อยู่ในรูปของ quadratic programming ได้แล้ว เราจะสามารถทำ relaxation ให้กลายเป็นปัญหาที่เรียกว่า semidefinite programming ซึ่งสามารถหาคำตอบได้ไม่ยาก อย่างไรก็ดีในปัญหา semidefinite programming นั้นเราจะมองตัวแปรที่ปรับค่าได้ในรูปของ matrix เราจึงควรจัดรูปปัญหาของเราให้เป็นปัญหาบน matrix ก่อน ซึ่งทำได้ง่ายเมื่อปัญหาที่เรามีอยู่ในรูป quadratic programming อยู่แล้ว

## relaxation สำหรับแบบจำลอง deep learning ที่มี 1 hidden layer
เพื่อความง่ายเราจะเริ่มพิจารณากรณีที่แบบจำลองของเรามี hidden layer เพียง layer เดียวก่อน ซึ่งจะสามารถเขียนในรูปปัญหา quadratic programming ได้ดังนี้

$$
\begin{array}{ll}
\max_{z_1,z_2} &(e_{y'}-e_y)W_2z_2\\
\text{subject to}\\
& z_1^2-(l+u)z_1+lu\leq 0\\
& z_2\geq W_1z_1+b_1\\
& z_2\geq 0\\
& z_2^2 -W_1z_1z_2 -b_1z_2 = 0
\end{array}
$$

ถ้าเรานิยามให้ $$v=\bigl[\begin{smallmatrix}
1\\
z_1\\
z_2
\end{smallmatrix}\bigr]$$ และให้ $$P=vv^T$$ เราจะใช้สัญลักษณ์ $$P[\cdot]$$ ในการระบุถึงสมาชิกของ $$P$$ ดังนี้

$$
P=\begin{bmatrix}
P[1] &P[z_1^T] & P[z_2^T]\\
P[z_1] & P[z_1z_1^T] & P[z_1z_2^T]\\
P[z_2] & P[z_2z_1^T] & P[z_2z_2^T]
\end{bmatrix}
$$

ถึงตรงนี้เราสามารถเขียนปัญหาของเราใหม่ในรูปของ $$P$$ ได้ดังนี้

$$
\begin{array}{ll}
\max_P & (e_{y'}-e_y)W_2P[z_2]\\
\text{subject to}&\\
& \text{diag}(P[z_1z_1^T])-(l+u)\cdot P[z_1]+l\cdot u\leq 0\\
& P[z_2]\geq W_1\cdot P[z_1]+b_1\\
& P[z_2]\geq 0\\
& \text{diag}(P[z_2z_2^T])-W_1\cdot \text{diag}(P[z_1z_2^T])-b_1\cdot P[z_2] = 0\\
& P[1] = 1\\
& P = vv^T \text{ for some vector } v
\end{array}
$$

เนื่องจากรูปแบบปัญหาล่าสุดนี้ยังคงเป็นปัญหาเดียวกับปัญหาตั้งต้นของเรา เราจึงยังไม่หวังว่าจะสามารถหาคำตอบได้อย่างมีประสิทธิภาพ อย่างไรก็ดี ปัญหาในรูปแบบล่าสุดนี้สามารถทำ relaxation ให้หาคำตอบได้ง่ายขึ้นโดยการเปลี่ยนเงื่อนไขที่กำหนดให้ $$P=vv^T$$ สำหรับเวกเตอร์ $$v$$ บางตัว ให้กลายเป็น $$P$$ สามารถอยู่ในรูปของ $$VV^T$$ เมื่อ $$V$$ เป็น matrix ที่มีจำนวน column เท่ากับจำนวนแถว เราเรียก symmetric matrix ที่มีคุณสมบัติดังกล่าวนี้ว่าเป็น _positive semidefinite matrix_ ถ้า $$P$$ เป็น positive semidefinite เราจะเขียนแทนว่า $$P\succeq 0$$ ดังนั้นเมื่อเรา relax เงื่อนไขนี้ เราก็จะได้ปัญหาที่เป็น semidefinite programming ดังนี้

$$
\begin{array}{ll}
\max_P & (e_{y'}-e_y)W_2P[z_2]\\
\text{subject to}&\\
& \text{diag}(P[z_1z_1^T])-(l+u)\cdot P[z_1]+l\cdot u\leq 0\\
& P[z_2]\geq W_1\cdot P[z_1]+b_1\\
& P[z_2]\geq 0\\
& \text{diag}(P[z_2z_2^T])-W_1\cdot \text{diag}(P[z_1z_2^T])-b_1\cdot P[z_2] = 0\\
& P[1] = 1\\
& P \succeq 0
\end{array}
$$

ซึ่งจะเห็นว่า กรณีที่มีเวกเตอร์ $$v$$ ที่ $$P=vv^T$$ นั้นก็เป็นหนึ่งในคำตอบที่เป็นไปได้ในปัญหาใหม่นี่เช่นกัน ดังนั้นปัญหาใหม่ของเราจึงครอบคลุมทุกความเป็นไปได้ของปั
ญหาตั้งต้น นั่นแสดงว่าค่า objective function ของปัญหาใหม่นี้ไม่มีทางน้อยกว่าค่า objective function ของปัญหาตั้งต้น แต่ปัญหา semidefinite programming นี้สามารถหาคำตอบที่ดีที่สุดได้ง่ายกว่ามาก การ relax เป็น semidefinite programming นี้จึงเป็นอีกเทคนิคที่น่าสนใจ

## relaxation สำหรับกรณีทั่วไป
คราวนี้เรามาพิจารณาเมื่อแบบจำลอง deep learning ของเรามีจำนวน layer เท่ากับ $$d$$ ซึ่งสามารถเขียนเงื่อนไขความสัมพันธ์ระหว่าง $$z_i$$ กับ $$z_{i+1}$$ ใด ๆ ได้เช่นเดียวกับความสัมพันธ์ระหว่าง $$z_1$$ กับ $$z_2$$ ในตัวอย่างที่ผ่านมา สมมติให้ทราบทราบขอบเขตของ $$z_i$$ ใด ๆ ว่า $$l_i\leq z_i\leq u_i$$ เราก็จะได้ปัญหา semidefinite programming ดังนี้

$$
\begin{array}{lll}
\max_P & (e_{y'}-e_y)W_dP[z_d]\\
\text{subject to}&\\
& \text{diag}(P[z_iz_i^T])-(l_i+u_i)\cdot P[z_i]+l_i\cdot u_i\leq 0, & \text{ for } i=1,\dots,d-1\\
& P[z_{i+1}]\geq W_i\cdot P[z_i]+b_i, & \text{ for } i=1,\dots,d-1\\
& P[z_{i+1}]\geq 0, & \text{ for } i=1,\dots,d-1\\
& \text{diag}(P[z_{i+1}z_{i+1}^T])-W_i\cdot \text{diag}(P[z_iz_{i+1}^T])-b_i\cdot P[z_{i+1}] = 0, & \text{ for } i=1,\dots,d-1\\
& P[1] = 1\\
& P \succeq 0
\end{array}
$$

โดยที่ขอบเขต  $$l_i$$ และ $$u_i$$ ก็สามารถใช้วิธีเดียวกับตอนทำ linear programming relaxation ในการหาได้

## References
1. [A. Raghunathan, J. Steinhardt, P. Liang. Semidefinite relaxations for certifying robustness to adversarial examples, In 32nd International Conference on Neural Information Processing Systems (NeurIPS), 2018](https://arxiv.org/abs/1811.01057)

---
Prev: [Quadratic programming](https://vacharapat.github.io/Adversarial-Machine-Learning/docs/cert3)

Next: [คุณสมบัติของ adversarial example](https://vacharapat.github.io/Adversarial-Machine-Learning/docs/feat1)
