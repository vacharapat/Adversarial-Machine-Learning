{% include lib/mathjax.html %}
# คุณสมบัติของ adversarial example

ในหัวข้อก่อนหน้านี้ เมื่อเราได้ค้นพบจุดอ่อนของแบบจำลอง machine learning ทั่วไปและสามารถสร้าง
adversarial example ได้ เราก็ได้หาวิธีการเพิ่มความทนทานของแบบจำลองต่อ adversarial example เหล่านี้ รวมไปถึงเทคนิคในการตรวจสอบความทนทานของแบบจำลองกันด้วย อย่างไรก็ดี สิ่งที่เราดูกันมานั้นยังไม่ได้อธิบายถึงสาเหตุความเปราะบางของแบบจำลอง และมีอยู่ของ adversarial example
เหล่านี้ว่าเกิดขึ้นได้อย่างไร

ในหัวข้อนี้เราจะมาทำความเข้าใจเกี่ยวกับสาเหตุของความเปราะบางของแบบจำลองและการเกิด adversarial example  โดยเริ่มจากการศึกษาคุณสมบัติของ
adversarial example ที่สร้างขึ้นมาได้

## การโจมตีข้ามแบบจำลอง


---
Prev: [Semidefinite programming relaxation](https://vacharapat.github.io/Adversarial-Machine-Learning/docs/cert4)

Next:
