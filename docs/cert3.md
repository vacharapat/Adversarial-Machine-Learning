{% include lib/mathjax.html %}
# Quadratic programming

ในหัวข้อนี้เราจะมาดูตัวอย่างการแปลงปัญหาของเราให้อยู่ในรูปแบบหนึ่งที่สามารถทำ relaxation ได้ไม่ยาก โดยรูปแบบของปัญหาที่เราจะจัดรูปนั้นเรียกว่า quadratic programming ซึ่งคือปัญหา optimization
ที่ objective function และ constrain ต่างอยู่ในรูป quadratic function บนตัวแปรที่ปรับค่าได้ (ฟังก์ชันที่ดีกรีของตัวแปรไม่เกิน 2)

เมื่อเราสามารถจัดรูปปัญหาให้เป็น quadratic programming ได้แล้ว เราจะสามารถทำ relaxation ให้กลายเป็นปัญหาที่เรียกว่า semidefinite programming ซึ่งจะได้ศึกษากันต่อไป

เราเริ่มต้นจากปัญหาตั้งต้นของเราดังนี้

$$
\begin{array}{ll}
\max_{z_1,\dots,z_{d+1}}& (e_{y'}-e_y)^Tz_{d+1}\\
\text{subject to}&\\
&\|z_1-x\|_\infty\leq\epsilon\\
&z_{i+1} = \max(0, W_iz_i+b_i), \text{ for } i=1,\dots, d-1\\
&z_{d+1}=W_dz_d+b_d
\end{array}
$$

ตอนที่เราแปลงปัญหานี้ให้อยู่ในรูป linear integer programming สังเกตว่าปัญหาใหม่ของเราจะมีจำนวนเงื่อนไขเพิ่มขึ้นค่อนข้างมาก
เนื่องจากแต่ละเงื่อนไขถูกบังคับให้อยู่ในรูปของ linear constrain เท่านั้น แต่ตอนนี้เราสามารถออกแบบเงื่อนไขในรูปของ quadratic constrain ได้
ซึ่งจะทำให้การแปลงปัญหาเป็น quadratic programming นั้นลดจำนวนเงื่อนไขลงจาก linear integer programming ได้

ลำดับแรก เงื่อนไข $$\|z_1-x\|_\infty\leq\epsilon$$ นั้นเราสามารถแปลงเป็น quadratic constrain เงื่อนไขเดียวได้
โดยถ้าเรากำหนดให้ $$l = x-\epsilon$$ และ $$u=x+\epsilon$$ เงื่อนไขของเราสามารถเขียนใหม่ได้เป็น $$l\leq z_1\leq u$$
ซึ่งจะเป็นจริงก็ต่อเมื่อ $$(z_1-l)(z_1-u)\leq 0$$
หรือเขียนใหม่เป็น quadratic constrain ได้ดังนี้

$$
z_1^2-(l+u)z_1+lu\leq 0
$$

สำหรับเงื่อนไข $$z_{i+1}=\max(0,W_iz_i+b_i)$$ นั้น เราสามารถเขียนในรูป quadratic constrain โดยใช้สามเงื่อนไขได้ดังนี้

$$
\begin{split}
z_{i+1}&\geq &W_iz_i+b_i\\
z_{i+1}&\geq &0\\
z_{i+1}(z_{i+1}-W_iz_i-b_i) &=& 0
\end{split}
$$

หากลองพิจารณาจะเห็นว่า สองเงื่อนไขแรกจะรับประกันว่าค่าของ $$z_{i+1}$$ จะต้องไม่น้อยไปกว่า $$\max(0,W_iz_i+b_i)$$ ในขณะที่เงื่อนไขที่สามแสดงว่า $$z_{i+1}$$ จะต้องมีค่าเท่ากับ 0 หรือไม่ก็เท่ากับ $$W_iz_i+b_i$$ ซึ่งตรงกับที่เราต้องการ เงื่อนไขที่สามเราอาจเขียนใหม่ได้เป็น

$$
z_{i+1}^2-W_iz_iz_{i+1}-b_iz_{i+1} =0
$$

ถึงตรงนี้เราสามารถเขียนปัญหาของเราอยู่ในรูป quadratic programming ได้แล้ว อย่างไรก็ดี เพื่อความสะดวกต่อไปเราจะตัดตัวแปร $$z_{d+1}$$ ออกโดยการนำเงื่อนไข $$z_{d+1}=W_dz_d+b_d$$ เข้าไปรวมใน objective function $$(e_{y'}-e_y)^Tz_{d+1}$$ แทน
ซึ่งจะได้ว่าฟังก์ชันที่เราต้องการให้มีค่าสูงที่สุดคือ

$$
(e_{y'}-e_y)^TW_dz_d - (e_{y'}-e_y)^Tb_d
$$

แต่เนื่องจาก $$(e_{y'}-e_y)^Tb_d$$ ไม่มีตัวแปรที่เราสามารถปรับค่าได้ ดังนั้น objective function ดังกล่าวจะมีค่าสูงที่สุดก็ต่อเมื่อค่าของ $$(e_{y'}-e_y)^TW_dz_d $$ ต้องสูงที่สุดเท่านั้น เมื่อเรานำเงื่อนไขทั้งหมดมารวม จะได้ว่าปัญหาของเราสามารถเขียนใหม่เป็น quadratic programming ได้ดังนี้

$$
\begin{array}{lll}
\max_{z_1,\dots,z_d} &(e_{y'}-e_y)^TW_dz_d\\
\text{subject to}\\
& z_1^2-(l+u)z_1+lu\leq 0\\
& z_{i+1}\geq W_iz_i+b_i,& \text{ for } i=1,\dots,d-1\\
& z_{i+1}\geq 0,& \text{ for } i=1,\dots,d-1\\
& z_{i+1}^2-W_iz_iz_{i+1}-b_iz_{i+1} =0,& \text{ for } i=1,\dots,d-1\\
\end{array}
$$

## References
1. [A. Raghunathan, J. Steinhardt, P. Liang. Semidefinite relaxations for certifying robustness to adversarial examples, In 32nd International Conference on Neural Information Processing Systems (NeurIPS), 2018](https://arxiv.org/abs/1811.01057)

---
Prev: [Linear programming relaxation](https://vacharapat.github.io/Adversarial-Machine-Learning/docs/cert2)

Next: [Semidefinite programming relaxation](https://vacharapat.github.io/Adversarial-Machine-Learning/docs/cert4)
