{% include lib/mathjax.html %}
# Adversarial robustness

เราได้เห็นตัวอย่างการโจมตีแบบจำลอง deep learning กันมาแล้ว คราวนี้เราจะมาพิจารณาในฝั่งของผู้สร้างแบบจำลองกันดูบ้างว่าเราจะสามารถสร้างแบบจำลองที่มี _ความทนทานต่อการโจมตี_ (adversarial robustness) เหล่านี้ได้อย่างไร โดยลำดับแรกเราจะมาทบทวนหัวใจหลักของการเทรนแบบจำลอง deep learning ในรูปแบบปกติกันก่อน

## Risk ของแบบจำลอง

สำหรับแบบจำลองทาง machine learning $$h_\theta$$ ใด ๆ เรานิยามให้ _risk_ ของแบบจำลองนี้เป็นค่าเฉลี่ยหรือ _ค่าคาดหวัง_ (expectation) ของ loss ดังนี้

$$
R(h_\theta) = \mathbb{E}_{(x,y)\sim \mathcal{D}}[\ell(h_\theta(x), y)]
$$

เมื่อ $$\mathcal{D}$$ เป็นการกระจายตัวของข้อมูลทั้งหมด ซึ่งในทางปฏิบัติเราไม่ทราบ $$\mathcal{D}$$ ที่แท้จริง เราจึงต้องทำการประมาณการกระจายตัวดังกล่าวจากเซตของตัวอย่างข้อมูล $$D=\{(x_1,y_1),\dots,(x_m,y_m)\}$$ ซึ่ง $$(x_i, y_i)$$ แต่ละตัวนั้นสุ่มมาจาก $$\mathcal{D}$$
ถ้าให้ $$\hat{R}(h_\theta,D)$$ แทนค่าประมาณของ $$R(h_\theta)$$ เมื่อใช้ $$D$$ เป็นตัวแทนการกระจายตัว เราสามารถคำนวณ $$\hat{R}(h_\theta,D)$$ ได้จาก

$$
\hat{R}(h_\theta,D)=\frac{1}{m}\sum_{i=1}^m\ell(h_\theta(x_i),y_i)
$$

ในกระบวนการเทรนแบบจำลองทาง machine learning โดยทั่วไป เราทำการหาค่าของพารามิเตอร์ $$\theta$$ ที่ทำให้ค่าประมาณของ risk  เมื่อพิจารณากับ training data $$D_\text{train}$$ มีค่าน้อยที่สุด นั่นคือ เราทำการแก้ปัญหา optimization ต่อไปนี้

$$
\min_\theta\hat{R}(h_\theta,D_\text{train})
$$

เนื่องจากเราเลือกพารามิเตอร์ $$\theta$$ ตามข้อมูลใน $$D_\text{train}$$ ข้อมูลชุดนี้จึงไม่สามารถนำมาใช้ในการวัดค่าประมาณของ risk อย่างเป็นกลางได้อีกต่อไป เราจึงมักจะต้องมีเซตของข้อมูลอีกชุดหนึ่งสำหรับวัด risk ของแบบจำลอง เราเรียกชุดข้อมูลดัวกล่าวว่าข้อมูลทดสอบ หรือ test data ($$D_\text{test}$$) ซึ่งสมาชิกได้มาจากการสุ่มจากการกระจายตัว $$\mathcal{D}$$ เช่นเดียวกัน เราจะใช้ $$\hat{R}(h_\theta,D_\text{test})$$ เป็นค่าประมาณของ risk $$R(h_\theta)$$ ที่แท้จริง

## Adversarial risk

จากหัวข้อที่แล้ว เราสามารถนำนิยามดั้งเดิมของ risk มาแก้ไขเพื่อให้สะท้อนความเสียหายที่เกิดจากการถูกก่อกวนด้วย adversarial attack ได้ โดยแทนที่เราจะสนใจ loss $$\ell(h_\theta(x), y)$$ โดยตรงสำหรับ sample $$(x,y)$$ เราพิจารณา loss ที่มากที่สุดที่อาจเกิดขึ้นจากการถูกก่อกวนบน sample $$(x,y)$$ แทน ดังนั้น เราสามารถนิยาม adversarial risk ได้เป็นค่าคาดหวังของ loss ดังกล่าว นั่นคือ เราจะให้ adversarial risk มีค่าเป็น

$$
R_\text{adv}(h_\theta)=\mathbb{E}_{(x,y)\sim\mathcal{D}}[\max_{\delta\in\Delta(x)}\ell(h_\theta(x+\delta),y)]
$$

เมื่อ $$\Delta(x)$$ เป็นเซตของการก่อกวนที่เป็นไปได้สำหรับ sample $$x$$ ซึ่งอาจแตกต่างกันตามแต่ละ sample ก็ได้ โดยที่เรายังคงต้องการให้การก่อกวนเหล่านี้ไม่สร้างความแตกต่างสำหรับมนุษย์เช่นเดียวกับตัวอย่างก่อนหน้า

จากนิยามดังกล่าว สังเกตว่าหากเรามีชุดของตัวอย่างข้อมูล $$D=\{(x_1,y_1),\dots,(x_m,y_m)\}$$ adversarial risk ที่ประมาณโดยใช้ $$D$$ เป็นตัวแทนการกระจายจะคำนวณได้จาก

$$
\hat{R}_\text{adv}(h_\theta,D)=\frac{1}{m}\sum_{i=1}^m\max_{\delta_i\in\Delta(x_i)}\ell(h_\theta(x_i+\delta_i),y_i)
$$

ดังนั้นจะเห็นว่า หากเรามี training data และต้องการเทรนแบบจำลองให้มีความทนทานต่อการโจมตีสูงแทนที่จะมีความแม่นยำสูงเพียงอย่างเดียว เราทำได้โดยหาทางกำหนดพารามิเตอร์ของแบบจำลองให้ adversarial risk เมื่อเทียบกับ training data นี้มีค่าน้อยที่สุดนั่นเอง

## References
1. [Z. Kolter, A. Madry. Adversarial Robustness - Theory and Practice](https://adversarial-ml-tutorial.org)

---
Prev: [การโจมตีแบบกำหนดเป้าหมาย](https://vacharapat.github.io/Adversarial-Machine-Learning/docs/intro3)

Next: [การสร้าง robust classifier](https://vacharapat.github.io/Adversarial-Machine-Learning/docs/intro5)
