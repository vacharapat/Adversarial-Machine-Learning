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
นั่นคือ เราต้องการให้ในการกระจายตัวของข้อมูล $$\widehat{D}_R$$ นั้น feature ใดที่ใช้ประกอบการตัดสินใจของ $C$
จะยังคงเป็น useful feature อยู่ในขณะที่ feature อื่น ๆ ใช้งานไม่ได้ เนื่องจากเราเชื่อว่า $C$ ที่มีความทนทานต่อการโจมตีนั้นจะตัดสินใจจาก robust feature เป็นหลัก เราจึงคาดว่าการกระจายตัว $$\widehat{D}_R$$
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

ภาพด้านล่างแสดงตัวอย่างของข้อมูลที่สร้างได้จากกระบวนการดังกล่าว

<p align="center">
<img width="500" src="https://raw.githubusercontent.com/vacharapat/Adversarial-Machine-Learning/master/images/frog.png">
</p>

## การเทรนแบบจำลองด้วย robust training set
เมื่อเราสร้าง training set ใหม่บนการกระจายตัว $$\widehat{D}_R$$ ได้แล้ว เราทดลองนำ training set
นี้มาเทรนแบบจำลองใหม่โดยใช้การเทรนแบบ standard training จากนั้นนำแบบจำลองที่เทรนได้ไปทดสอบกับ
test set ตั้งต้น (บนการกระจายตัว $D$) ผลที่ได้เป็นดังภาพด้านล่าง ซึ่งแสดงให้เห็นว่าแบบจำลองที่ได้นั้นมีความแม่นยำสูงและยังมีความทนทานที่ดีขึ้นกว่าการทำ standard training
บนชุดข้อมูลดั้งเดิมอีกด้วย

<p align="center">
<img width="550" src="https://raw.githubusercontent.com/vacharapat/Adversarial-Machine-Learning/master/images/CIFAR_res.png">
</p>

คราวนี้หากเราทำการทดลองแบบเดิม โดยที่เปลี่ยนแบบจำลอง $C$ ที่ใช้ประกอบการสร้างชุดข้อมูลใหม่ให้เป็นแบบจำลองที่สร้างจาก standard training (ซึ่งไม่มีความทนทานต่อการโจมตี)
ให้ $$\widehat{D}_{NR}$$ แทนชุดข้อมูลที่ได้จากการสร้างด้วย $C$ อย่างแรกที่เราเห็นคือ ตัวอย่างข้อมูลใน
$$\widehat{D}_{NR}$$ นั้นมีความเป็น noise อย่างมากและไม่มีลักษณะใกล้เคียงกับภาพตั้งต้นที่ใช้ ซึ่งผิดกับตัวอย่างข้อมูลใน $$\widehat{D}_R$$ ที่ดึงลักษณะเด่นของภาพตั้งต้นออกมาได้มาก

ถัดมา เมื่อเรานำชุดข้อมูล $$\widehat{D}_{NR}$$ มาเทรนแบบจำลองใหม่ด้วยการทำ standard training
และทำการทดสอบแบบจำลองที่ได้บน test set ดั้งเดิม (บนการกระจายตัว $D$) เราพบว่าแบบจำลองที่ได้นั้นมีความแม่นยำสูง
ถึงแม้ว่าข้อมูลที่ใช้เทรนมีลักษณะเป็น noise อย่างมาก อย่างไรก็ดีเมื่อทดสอบความแม่นยำกับ adversarial example
เราพบว่าแบบจำลองที่ได้นี้ไม่มีความทนทานต่อการโจมตี ภาพด้านล่างสรุปภาพรวมของการทดลองที่กล่าวมาทั้งหมด

<p align="center">
<img width="500" src="https://raw.githubusercontent.com/vacharapat/Adversarial-Machine-Learning/master/images/robust_nonrobust_dataset.png">
</p>

ผลจากการทดลองนี้สนับสนุนแนวคิดที่ว่า weak feature ในข้อมูลนั้นมีอยู่จริง และในการตัดสินใจของแบบจำลองที่เทรนแบบ standard training นั้นมีการใช้ประโยชน์จาก weak feature ประกอบเป็นอย่างมาก (ดูจากตัวอย่างภาพที่สร้างได้จาก classifier $C$ ที่เป็น standard model ซึ่งเราไม่เห็นคุณลักษณะเด่นของคลาสเลย นั่นแสดงว่าตัวอย่างข้อมูลนี้ดึงคุณลักษณะของ weak feature ขึ้นมามากนั่นเอง)
นอกจากนี้การที่ชุดข้อมูลที่สร้างจาก robust model สามารถเทรนแบบจำลองใหม่ให้มีความทนทานได้โดยใช้ standard training ก็ยังสนับสนุนแนวคิดที่ว่าตัว robust model ที่เทรนด้วย adversarial training นั้นทำการตัดสินใจโดยใช้
robust feature เป็นหลัก

## References

1. [A. Ilyas, S. Santurkar, D. Tsipras, L. Engstrom, B. Tran, A. Madry. Adversarial Examples Are Not Bugs, They Are Features, In: Advances in Neural Information Processing Systems, 2019](https://arxiv.org/abs/1905.02175)

---
Prev: [Robust features และ weak features](https://vacharapat.github.io/Adversarial-Machine-Learning/docs/feat3)

Next: [ความสามารถของ weak features](https://vacharapat.github.io/Adversarial-Machine-Learning/docs/feat5)
