{% include lib/mathjax.html %}
# Robust features และ weak features

ในบรรดา useful feature ที่แบบจำลองใช้ในการจำแนกข้อมูลนั้น อาจจะมีบาง feature ที่สอดคล้องกับที่มนุษย์สังเกตได้
เช่นหากข้อมูลเป็นภาพ เราสามารถจำแนกภาพสิ่งมีชีิตได้โดยสังเกตว่าในภาพนั้นมีดวงตา หรือมีใบหู เป็นต้น
อย่างไรก็ดี การที่ adversarial example สามารถทำให้แบบจำลองเปลี่ยนการตัดสินใจเป็นคลาสอื่นได้ทั้ง ๆ
ที่ในสายตามนุษย์ไม่เห็นความแตกต่างจากตัวอย่างข้อมูลเดิมนั้น แสดงว่าน่าจะต้องมี feature บางอย่างที่มีส่วนร่วมในการตัดสินใจของแบบจำลอง ที่ลักษณะของ feature ดังกล่าวเปลี่ยนไปจากเดิมอย่างชัดเจนใน
adversarial example อย่างไรก็ดี เนื่องจากในสายตามนุษย์เรานั้นไม่เห็นความแตกต่างจากข้อมูลเดิม
แสดงว่าเราไม่สามารถสังเกตเห็น feature ที่เปลี่ยนแปลงไปเหล่านี้ได้ ตรงนี้ทำให้เรามีแนวทางในการแยกแยะระหว่าง
useful feature ที่มนุษย์สังเกตได้และสังเกตไม่ได้ออกจากกันด้วยนิยามต่อไปนี้

## Robust features
สำหรับ useful feature ที่มนุษย์สังเกตได้นั้น ควรจะต้องมีการเปลี่ยนแปลงจากเดิมไม่มากใน adversarial example
กล่าวคือ feature ในกลุ่มนี้ยังคงเป็น useful feature ภายใต้การโจมตีด้วย
อย่างไรก็ดีเราไม่สามารถรับประกันว่า feature ที่สอดคล้องกับนิยามนี้จะสามารถสังเกตได้ในสายตามนุษย์เสมอไป
เพื่อความชัดเจนเราจะเรียก feature เหล่านี้ว่าเป็น _robust feature_ กล่าวคือ

สำหรับ useful feature $f\in F$ ใด ๆ $f$ จะเป็น robust feature ถ้า

$$
\mathbb{E}_{(x,y)\sim D}\left[y\cdot \inf_{\delta\in\Delta(x)}f(x+\delta)\right]\geq\rho'
$$

เมื่อ $\rho'>0$

## Weak features
สำหรับ feature $f$ ที่เป็น useful feature บนค่า $\rho>0$ บางค่า แต่ไม่เป็น robust feature
บนค่า $\rho'>0$ ใด ๆ เลย เราจะเรียก $f$ ว่าเป็น _weak feature_

สังเกตว่า weak feature นั้นมีส่วนช่วยในการตัดสินใจของแบบจำลองได้ดีเมื่อเราทำการเรียนรู้แบบทั่วไป
แต่เป็นตัวที่สร้างปัญหาเมื่อเจอกับ adversarial example เนื่องจาก feature กลุ่มนี้ไม่ได้ให้คะแนนไปทางคลาสเดิมอีกต่อไป

จากนี้ยามของ feature สองกลุ่มนี้ เราจะมาดูการทดลองเพื่อวิเคราะห์บทบาทของ robust feature และ weak feature ในการตัดสินใจของแบบจำลองกันต่อไป



## References

1. [I. Goodfellow, J. Shlens, C. Szegedy. Explaining and Harnessing Adversarial Examples,
In Intenational Conference on Learning Representations (ICLR), 2015](https://arxiv.org/abs/1412.6572)

---
Prev: [Useful features](https://vacharapat.github.io/Adversarial-Machine-Learning/docs/feat2)

Next:
