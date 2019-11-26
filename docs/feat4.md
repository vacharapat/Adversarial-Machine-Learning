{% include lib/mathjax.html %}
# การเทรนแบบจำลองด้วย robust features

จากกระบวนการเทรนแบบจำลองแบบปกติ (standard training) นั้น การที่ feature $f$ ใด ๆ จะถูกใช้เป็นองค์ประกอบในการตัดสินใจของแบบจำลองจะขึ้นอยู่กับว่า
$f$ เป็น useful feature หรือไม่ ดังนั้นจากแนวคิดเรื่อง robust feature แสดงว่าถ้าเราสามารถรับประกันได้ว่า
useful feature ทั้งหมดในการกระจายตัวของข้อมูลของเราเป็น robust feature ทั้งสิ้น
การเทรนแบบจำลองแบบ standard training ก็ควรที่จะให้ผลลัพธ์เป็น classifier ที่มีความทนทานต่อการโจมตี

อย่างไรก็ดี เราไม่สามารถเข้าไปจัดการแก้ไข feature ต่าง ๆ ในชุดข้อมูลของเราโดยตรงให้ได้ผลตามต้องการได้
เนื่องจากชุดข้อมูลมีความซับซ้อนและ dimension ที่สูงมาก เทคนิคที่เราทำได้คือ
เราจะทำการปรับปรุงชุดข้อมูลให้เหลือเฉพาะ robust feature เท่านั้นที่ใช้ประโยชน์ได้ โดยใช้แบบจำลองที่มีความทนทานอยู่แล้วเข้ามาช่วย

## Robust training set
กำหนดให้ $C$ เป็นแบบจำลองที่ทนทานต่อการโจมตี (เช่นแบบจำลองที่ถูกเทรนด้วย [adversarial training](https://vacharapat.github.io/Adversarial-Machine-Learning/docs/attack5))
และให้ $D$ เป็นการกระจายตัวของข้อมูลที่เราต้องการเรียนรู้ เราต้องการที่จะสร้างการกระจายตัวของข้อมูล
$\widehat{D}_R$ ที่มีคุณสมบัติดังนี้

$$
\mathbb{E}_{(x,y)\sim \widehat{D}_R}[y\cdot f(x)] =
\begin{cases}
\mathbb{E}_{(x,y)\sim D}[y\cdot f(x)] & \text{ ถ้า } f\in F_C\\
0 & \text{ กรณีอื่น}
\end{cases}
$$

โดยที่ $F_C$ แทนเซตของ feature ที่ใช้ประกอบในการตัดสินใจของ classifier $C$
นั่นคือ เราต้องการให้ในการกระจายตัวของข้อมูล $\widehat{D}_R$ นั้น feature ใดที่ใช้ประกอบการตัดสินใจของ $C$
จะยังคงเป็น useful feature อยู่ในขณะที่ feature อื่น ๆ ใช้งานไม่ได้ เนื่องจากเราเชื่อว่า $C$ ที่มีความทนทานต่อการโจมตีนั้นจะตัดสินใจจาก robust feature เป็นหลัก เราจึงคาดว่าการกระจายตัว $\widehat{D}_R$
ที่จะได้มานั้นจะมีเฉพาะ robust feature ที่เป็น useful feature เราจะเรียก training set ที่การกระจายตัวมีคุณสมบัติเช่นนี้ว่าเป็น _robust training set_

ในการสร้าง training set ของ $\widehat{D}_R$ เราจะพิจารณาตัวอย่างข้อมูล $x$ แต่ละตัวใน
training set เดิมบนการกระจายตัว $D$ และทำการสร้างตัวอย่างข้อมูล $x_r$ ขึ้นมาใหม่โดยอิงจาก $x$ และ $C$ ดังนี้
เราเริ่มจากการสุ่ม noise $x_0$ และใช้ gradient descent ในการหา $x_r$ ที่

$$
\min_{x_r\in\mathcal{X}}\|h(x_r) - h(x)\|_2
$$

เมื่อ $h$ เป็นฟังก์ชันที่ map จาก input $x$ ไปยัง representation layer สังเกตว่าเราเริ่มจากการสุ่ม $x_0$
เพื่อให้ feature ใด ๆ ไม่สามารถเป็น useful feature ได้ จากนั้นเราพยายามปรับปรุง input จาก
$x_0$ ให้ใกล้เคียงกับ $x$ มากขึ้นเรื่อย ๆ ผ่านสายตาของ $C$ ถ้าหาก $C$ ใช้เพียง robust feature ในการตัดสินใจ
เราจะได้ว่าการปรับปรุง input นี้จะทำการปรับปรุงเฉพาะ robust feature ให้เข้าไปใกล้เคียงกับ robust feature
ในตัวอย่างข้อมูล $x$ เดิมให้ได้มากที่สุด กระบวนการสร้าง robust training set สามารถเขียนเป็นรหัสลำลองได้ดังนี้

$$
\begin{array}{l}
\text{RobustDataset($D$):}\\
\quad C\gets \text{AdversarialTraining($D$)}\\
\quad h\gets \text{mapping ของ $C$ จาก input ไปยัง representation layer}\\
\quad D_R\gets\emptyset\\
\quad \text{for } (x,y)\in D:\\
\quad \quad x_0\gets \text{ random noise}\\
\quad \quad x_r\gets \arg\min_{x_r\in\mathcal{X}}\|h(x_r) - h(x)\|_2 \quad\text{(ใช้ PGD โดยเริ่มต้นที่ $x_0$)}\\
\quad \quad D_R\gets D_R\cup \{(x_r, y)\}\\
\quad \text{Return } D_R
\end{array}
$$

## References

1. [A. Ilyas, S. Santurkar, D. Tsipras, L. Engstrom, B. Tran, A. Madry. Adversarial Examples Are Not Bugs, They Are Features, In: Advances in Neural Information Processing Systems, 2019](https://arxiv.org/abs/1905.02175)

---
Prev: [Robust features และ weak features](https://vacharapat.github.io/Adversarial-Machine-Learning/docs/feat3)

Next:
