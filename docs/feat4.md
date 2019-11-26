{% include lib/mathjax.html %}
# การเทรนแบบจำลองด้วย robust features

จากกระบวนการเทรนแบบจำลองแบบปกติ (standard training) นั้น การที่ feature $f$ ใด ๆ จะถูกใช้เป็นองค์ประกอบในการตัดสินใจของแบบจำลองจะขึ้นอยู่กับว่า
$f$ เป็น useful feature หรือไม่ ดังนั้นจากแนวคิดเรื่อง robust feature แสดงว่าถ้าเราสามารถรับประกันได้ว่า
useful feature ทั้งหมดในการกระจายตัวของข้อมูลของเราเป็น robust feature ทั้งสิ้น
การเทรนแบบจำลองแบบ standard training ก็ควรที่จะให้ผลลัพธ์เป็น classifier ที่มีความทนทานต่อการโจมตี

อย่างไรก็ดี เราไม่สามารถเข้าไปจัดการแก้ไข feature ต่าง ๆ ในชุดข้อมูลของเราโดยตรงให้ได้ผลตามต้องการได้
เนื่องจากชุดข้อมูลมีความซับซ้อนและ dimension ที่สูงมาก เทคนิคที่เราทำได้คือ
เราจะทำการปรับปรุงชุดข้อมูลให้เหลือเฉพาะ robust feature เท่านั้นที่ใช้ประโยชน์ได้ โดยใช้แบบจำลองที่มีความทนทานอยู่แล้วเข้ามาช่วย

# Robust training set
กำหนดให้ $C$ เป็นแบบจำลองที่ทนทานต่อการโจมตี (เช่นแบบจำลองที่ถูกเทรนด้วย [adversarial training](https://vacharapat.github.io/Adversarial-Machine-Learning/docs/attack5))
และให้ $D$ เป็นการกระจายตัวของข้อมูลที่เราต้องการเรียนรู้ เราต้องการที่จะสร้างการกระจายตัวของข้อมูล
$\widehat{D}_R$ ที่มีคุณสมบัติดังนี้

$$
\mathbb{E}_{(x,y)\sim \widehat{D}_R}[y\cdot f(x)] =
\begin{cases}
\mathbb{E}_{(x,y)\sim D}[y\cdot f(x)] & \text{ ถ้า } f\in F_c\\
0 & \text{ กรณีอื่น}
\end{cases}
$$

## References

1. [A. Ilyas, S. Santurkar, D. Tsipras, L. Engstrom, B. Tran, A. Madry. Adversarial Examples Are Not Bugs, They Are Features, In: Advances in Neural Information Processing Systems, 2019](https://arxiv.org/abs/1905.02175)

---
Prev: [Robust features และ weak features](https://vacharapat.github.io/Adversarial-Machine-Learning/docs/feat3)

Next:
