{% include lib/mathjax.html %}
#  การสร้าง robust classifier

เมื่อเรามีนิยามของ adversarial risk อย่างชัดเจนแล้ว ปัญหาในการเทรนแบบจำลองให้มีความทนทานสูงก็นิยามได้อย่างตรงไปตรงมา นั่นคือ สำหรับ training data $$D_\text{train}=\{(x_1,y_1),\dots,(x_m,y_m)\}$$ เราต้องการหาพารามิเตอร์ $$\theta$$ ที่เป็นตำตอบของปัญหา optimization ต่อไปนี้

$$
\min_\theta\hat{R}(h_\theta,D_\text{train}) = \min_\theta\frac{1}{m}\sum_{i=1}^m\max_{\delta_i\in\Delta(x_i)}\ell(h_\theta(x_i+\delta_i),y_i)
$$

เราจะเรียก formulation นี้ว่าเป็น min-max หรือ robust optimization formulation สำหรับการทำ adversarial learning ซึ่งจะถูกกล่าวถึงบ่อย ๆ ต่อไป

วิธีหนึ่งในการแก้ปัญหานี้ ก็คือการใช้อัลกอริทึมการเทรนเช่นเดียวกับการเทรนแบบดั้งเดิม นั่นคือเราสามารถใช้ stochastic gradient descent สำหรับปรับค่า $$\theta$$ โดยในแต่ระรอบ เราทำการเลือก minibatch $$B\subseteq D_\text{train}$$ และทำการ update ค่าของ $$\theta$$ ในทิศทางตรงข้ามกับ gradient ดังนั้น

$$
\theta_{t+1}=\theta_t - \alpha\cdot\frac{1}{|B|}\sum_{(x,y)\in B}\nabla_\theta\max_{\delta\in\Delta(x)}\ell(h_\theta(x+\delta),y)
$$

จะเห็นว่าข้อแตกต่างสำคัญระหว่างการทำ adversarial training กับการ traing แบบดั้งเดิมนั้นก็คือ ในกรณีของ adversarial training การคำนวณ gradient ตามต้องการนี้ไม่ใช่เรื่องง่าย เนื่องจากตัวฟังก์ชันที่สนใจนั้นมีปัญหา maximization อยู่ภายใน อย่างไรก็ดี [Danskin's theorem](https://en.wikipedia.org/wiki/Danskin's_theorem) ได้แสดงไว้ว่าเราสามารถหา gradient ของฟังก์ชัน max ได้จากการคำนวณ gradient ของฟังก์ชันที่อยู่ข้างใน max โดยใช้ค่าของตัวแปรที่ทำให้ฟังก์ชันภายในนี้มีค่ามากที่สุด นั่นคือ

$$
\frac{\partial}{\partial v}\max_{u\in U}f(u,v) =\frac{\partial}{\partial v}f(u^*,v)
$$

เมื่อ 

$$
u^*=\arg\max_{u\in U}f(u,v)
$$

ดังนั้น จะเห็นว่าเราสามารถคำนวณ gradient ของ adversarial loss สำหรับตัวอย่างข้อมูล $$(x,y)$$  ได้โดยเริ่มจากการหาค่า $$\delta^*$$ ที่เป็นคำตอบของปัญหา maximization ข้างในเสียก่อน นั่นคือ ถ้าเราให้

$$
\delta^*=\arg\max_{\delta\in\Delta(x)}\ell(h_\theta(x+\delta),y)
$$

เราจะสามารถคำนวณ gradient ของ loss ของข้อมูล $$(x,y)$$ ได้จาก

$$
\nabla_\theta\max_{\delta\in\Delta(x)}\ell(h_\theta(x+\delta),y) = \nabla_\theta\ell(h_\theta(x+\delta^*),y)
$$

ถึงตรงนี้เราจะได้ว่า ในกระบวนการเทรน robust classifier ด้วย gradient descent การปรับพารามิเตอร์ในแต่ละรอบจะทำได้ในสองขั้นตอนคือ

1. สำหรับแต่ละตัวอย่างข้อมูล $$(x,y)\in B$$ แก้ปัญหา maximization ภายใน นั่นคือ หา

    $$\delta^*(x)=\arg\max_{\delta\in\Delta(x)}\ell(h_\theta(x+\delta),y)$$

2. คำนวณ gradient ของ adversarial risk และทำการปรับ $$\theta$$ ตาม

    $$\theta_{t+1}=\theta_t-\alpha\cdot\frac{1}{|B|}\sum_{(x,y)\in B}\nabla_\theta\ell(h_\theta(x+\delta^*(x)),y)$$

เราอาจมองว่าในการเทรนแต่ละรอบ เราเริ่มจากการหา adversarial example ก่อน จากนั้นก็ทำการปรับพารามิเตอร์ของแบบจำลองโดยอิงกับ adversarial example เหล่านี้แทนที่จะอิงกับตัว original data 

การทำ adversarial training ในลักษณะนี้มีความท้าทายอยู่ที่การจะแก้ปัญหา maximization ข้างในนั้นไม่ใช่เรื่องง่าย เนื่องจากสำหรับแบบจำลอง deep learning นั้น ปัญหา maximization ดังกล่าวไม่ได้อยู่ในกลุ่ม convex optimization ที่จะหาคำตอบได้เร็ว การใช้ gradient descent แบบที่ได้แสดงในการทดลองตอนแรกนั้นนอกจากจะเสียเวลามากสำหรับแต่ละ sample แล้วยังสามารถหาได้เพียงคำตอบที่เป็น local optimum ซึ่งหากเรานำคำตอบเหล่านี้มาใช้ในการคำนวณ gradient ในขั้นตอนที่สอง ก็ไม่สามารถรับประกันได้ว่าเราจะทำการปรับพารามิเตอร์ไปในทิศทางที่ถูกต้อง เนื่องจาก Danskin's theorem นั้น apply ได้สำหรับคำตอบที่นำมาสู่ค่า maximum จริงเท่านั้น อย่างไรก็ดี ในทางปฏิบัติแล้วเราพบว่าถ้าเราสามารถแก้ปัญหา maxmimization ข้างในได้ดีมากพอ (อาจไม่ต้องดีที่สุด) อัลกอริทึมนี้ก็ให้ผลที่ดีได้ นี่เป็นที่มาของอัลกอริทึมกลุ่มหนึ่งสำหรับประมาณคำตอบของปัญหา maximization ข้างในอย่างรวดเร็วที่เราจะได้เห็นกันต่อไป
