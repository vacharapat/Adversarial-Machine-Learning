{% include lib/mathjax.html %}
# Useful features

เนื่องจาก adversarial example ที่สร้างจาก FGSM หรือ PGD นั้นมีลักษณะใกล้เคียงกับตัวอย่างข้อมูลเดิมจนในบางครั้งอาจมองไม่เห็นความแตกต่างในสายตามนุษย์ การที่แบบจำลอง machine learning ทำนายคลาสของ adversarial example ผิดไปจากเดิมด้วยความมั่นใจที่สูงได้นั้นแสดงว่าแบบจำลองจะต้อง
_มองเห็น_ ความแตกต่างบางอย่างที่มนุษย์เรามองไม่เห็น โดยที่ความแตกต่างดังกล่าวมีผลต่อการตัดสินใจของแบบจำลองอย่างมาก ในหัวข้อนี้เราจะมาศึกษาเกี่ยวกับ feature
ของข้อมูลทั้งที่มนุษย์สังเกตได้และสังเกตไม่ได้ และทดสอบบทบาทของ feature
ทั้งสองกลุ่มนี้ที่มีต่อการตัดสินใจของแบบจำลอง

## Features
เพื่อความสะดวกในขั้นต้น เราพิจารณาแบบจำลองสำหรับจำแนกข้อมูลเป็นสองคลาส โดยที่ตัวอย่างข้อมูล
$$(x, y)\in \mathcal{X} \times\{-1, +1\}$$ ถูกหยิบมาจากการกระจายตัว $D$ เป้าหมายของเราคือการเรียนรู้เพื่อสร้าง
classifier $$C:\mathcal{X}\to\{-1, +1\}$$ สำหรับทำนายคลาส $y$ จาก input $x$

เราจะนิยามให้ _feature_ เป็นฟังก์ชันจาก input space $\mathcal{X}$ ไปยังจำนวนจริง $\mathbb{R}$
โดยเราสามารถมอง feature หนึ่งในรูปของปริมาณหรือค่าของคุณสมบัติหนึ่งที่วัดได้จาก input $x$ เช่น
feature $f_1$ วัดการมีอยู่ของดวงตาในรูป $x$ หรือ feature $f_2$ วัดการมีอยู่ของล้อรถในรูป $x$

ให้ $$\mathcal{F}=\{f:\mathcal{X}\to \mathbb{R}\}$$ เป็นเซตของ feature ทั้งหมดที่แบบจำลองสามารถสกัดได้ เพื่อความสะดวกเราสมมติให้ feature ทั้งหมดนี้มีค่าเฉลี่ยเป็น 0 และมีค่าความแปรปรวน (variance) เป็น 1
นั่นคือ

$$
\mathbb{E}_{(x,y)\sim D}[f(x)] = 0
$$  

และ

$$
\mathbb{E}_{(x,y)\sim D}[f(x)^2]=1
$$

สำหรับทุก feature $$f\in \mathcal{F}$$

เราจะเริ่มให้นิยามเกี่ยวกับ feature ที่มีบทบาทในการตัดสินใจของแบบจำลองทาง machine learning ดังนี้

## Useful features
สำหรับการกระจายตัว $D$ เราจะกล่าวว่า feature $f$ นั้น $\rho$-useful สำหรับค่าคงที่ $\rho>0$
ถ้า $f$ นั้นสอดคล้องกับคลาสที่ถูกต้องไม่น้อยกว่า $\rho$ โดยเฉลี่ย กล่าวคือ

$$
\mathbb{E}_{(x,y)\sim D}[y\cdot f(x)]\geq\rho
$$

โดยถ้าหาก feature $f$ นั้นแปรผกผันกับค่าของ $y$ เราจะถือว่า $-f$ เป็น useful feature

กำหนดให้แบบจำลองของเรานั้นมีกระบวนการตัดสินใจโดยดูจาก linear combination
ของ feature ทั้งหมดที่วัดได้จากข้อมูล input กล่าวคือ classifier $C$ จะทำนาย label $y$ ของ
input $x$ ดังนี้

$$
C(x) = \text{sgn}\left(\sum_{f\in F}w_f\cdot f(x) + b\right)
$$

เมื่อ $F$ เป็นเซตของ feature ทั้งหมด $w_f$ เป็น weight ของ feature $f$ ใน linear combination
และมี $b$ เป็นค่า bias

สังเกตว่าในขั้นตอนการเรียนรู้นั้น เราจะทำการหาค่า $w_f$ สำหรับ feature $f$ ต่าง ๆ และค่า bias $b$
ที่ทำให้ loss function ที่เกิดจากการทำนายผิดโดยเฉลี่ยมีค่าน้อยที่สุด หากเรามองผลของ linear combination ของ classifier (ก่อนที่จะเข้าฟังก์ชัน sgn เพื่อจำแนกคลาส) เป็นคะแนน จะเห็นว่าในการเทรนแบบจำลองนั้นจะทำให้ค่าเฉลี่ยของคะแนนที่ตอบถูกสูงที่สุด หรือกล่าวได้ว่าในการเทรนแบบจำลองนั้น
เราสามารถมองเป็นการ maximize

$$
\mathbb{E}_{(x,y)\sim D}\left[y\cdot \left(\sum_{f\in F}w_f\cdot f(x) + b\right)\right]
$$




## References

1. [A. Ilyas, S. Santurkar, D. Tsipras, L. Engstrom, B. Tran, A. Madry. Adversarial Examples Are Not Bugs, They Are Features, In: Advances in Neural Information Processing Systems, 2019](https://arxiv.org/abs/1905.02175)

---
Prev: [คุณสมบัติของ adversarial example](https://vacharapat.github.io/Adversarial-Machine-Learning/docs/feat1)

Next:
