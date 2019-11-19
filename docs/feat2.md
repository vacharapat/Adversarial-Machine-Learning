{% include lib/mathjax.html %}
# Useful features และ robust features

เนื่องจาก adversarial example ที่สร้างจาก FGSM หรือ PGD นั้นมีลักษณะใกล้เคียงกับตัวอย่างข้อมูลเดิมจนอาจมองไม่เห็นความแตกต่างในสายตามนุษย์ การที่แบบจำลอง machine learning ทำนายคลาสของ adversarial example ผิดไปจากเดิมด้วยความมั่นใจที่สูงได้นั้นแสดงว่าแบบจำลองจะต้อง
_มองเห็น_ ความแตกต่างบางอย่างที่มนุษย์เรามองไม่เห็น โดยที่ความแตกต่างดังกล่าวมีผลต่อการตัดสินใจของแบบจำลองอย่างมาก ในหัวข้อนี้เราจะมาศึกษาเกี่ยวกับ feature
ของข้อมูลทั้งที่มนุษย์สังเกตได้และสังเกตไม่ได้ และทดสอบบทบาทของ feature ทั้งสองกลุ่มนี้ในการตัดสินใจของแบบจำลอง

## Features
เพื่อความสะดวกในขั้นต้น เราพิจารณาแบบจำลองสำหรับจำแนกข้อมูลเป็นสองคลาส โดยที่ตัวอย่างข้อมูล
$$(x, y)\in \mathcal{X} \times\{-1, +1\}$$ ถูกหยิบมาจากการกระจายตัว $D$ เป้าหมายของเราคือการเรียนรู้เพื่อสร้าง
classifier $$C:\mathcal{X}\to\{-1, +1\}$$ สำหรับทำนายคลาส $y$ จาก input $x$

เราจะนิยามให้ _feature_ เป็นฟังก์ชันจาก input space $\mathcal{X}$ ไปยังจำนวนจริง $\mathbb{R}$

## References

1. [A. Ilyas, S. Santurkar, D. Tsipras, L. Engstrom, B. Tran, A. Madry. Adversarial Examples Are Not Bugs, They Are Features, In: Advances in Neural Information Processing Systems, 2019](https://arxiv.org/abs/1905.02175)

---
Prev: [คุณสมบัติของ adversarial example](https://vacharapat.github.io/Adversarial-Machine-Learning/docs/feat1)

Next:
