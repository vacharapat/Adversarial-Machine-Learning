{% include lib/mathjax.html %}
# ความสามารถของ weak feature

หลังจากที่เราได้เห็นความสามารถในการตัดสินใจของ robust feature กันมาแล้ว
คราวนี้เราจะมาลองดูความสามารถของ weak feature กันดูบ้าง โดยเราจะสร้างชุดข้อมูลใหม่ที่ทำให้
robust feature ทั้งหมดไม่สามารถใช้ช่วยในการตัดสินใจได้ นั่นคือเราจะทำให้ในชุดข้อมูลใหม่นั้นมีเพียง
weak feature ที่ยังคงเป็น useful feature อยู่ จากนั้นเราจะนำชุดข้อมูลที่ได้ไปเทรนแบบจำลองและนำมาทดสอบกับ
test data เดิมของเราดูเช่นเดียวกับการทดสอบความสามารถของ robust feature ก่อนหน้านี้

## การสร้าง weak-feature training set

ในการสร้างชุดข้อมูลให้มีเฉพาะ weak feature เท่านั้นที่เป็น useful feature เราเริ่มจากการสร้าง
standard classifier $C$ ซึ่งไม่มีความทนทานต่อการโจมตี จากนั้น สำหรับตัวอย่างข้อมูล $(x,y)$ แต่ละตัวใน
training set เราจะสร้างตัวอย่างข้อมูลใหม่ $(x',y')$ ด้วยวิธีการดังนี้

เราจะสุ่มคลาสเป้าหมาย $y'$ จากคลาสทั้งหมดให้มีความน่าจะเป็นที่จะได้แต่ละคลาสเท่ากัน
จากนั้นเราจะทำการโจมตีแบบกำหนดเป้าหมายเพื่อหา adversarial example $x'$ ที่ $C$ ทำนายว่าเป็นคลาส $y'$
โดยให้การก่อกวนมีขนาดเล็ก ๆ (ไม่เกิน $\epsilon$) การหา adversarial example $x'$ ดังกล่าวทำได้โดยการทำ
gradient descent เพื่อหา

$$
x' = {\arg\min}_{\|x'-x\|\leq\epsilon}\mathcal{L}_C(x', y')
$$



## References

1. [I. Goodfellow, J. Shlens, C. Szegedy. Explaining and Harnessing Adversarial Examples,
In Intenational Conference on Learning Representations (ICLR), 2015](https://arxiv.org/abs/1412.6572)

---
Prev: [การเทรนแบบจำลองด้วย robust features](https://vacharapat.github.io/Adversarial-Machine-Learning/docs/feat4)

Next:
