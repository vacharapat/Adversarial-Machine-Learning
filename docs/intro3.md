{% include lib/mathjax.html %}
# การโจมตีแบบกำหนดเป้าหมาย

จากตัวอย่างการทดลองในหัวข้อก่อนหน้านี้ เนื่องจากวอมแบตก็ไม่ได้ดูแตกต่างกับหมูมากมายนัก ความผิดพลาดดังกล่าวอาจดูไม่ใช่ปัญหาใหญ่
ในหัวข้อนี้เราจะแสดงให้เห็นว่าด้วยเทคนิคเดียวกันนี้ เราสามารถก่อกวนให้รูปหมูถูกทำนายเป็น class ใดก็ได้ตามที่เราต้องการ
เราเรียกการโจมตีลักษณะนี้ว่า _การโจมตีแบบกำหนดเป้าหมาย_ (targeted attack) โดยมีวิธีการคือ แทนที่เราจะหา $$\delta\in \Delta$$
ที่ทำให้ loss ของคลาสที่แท้จริงสูงที่สุดเพียงอย่างเดียว เราจะพยายามทำให้ loss ของคลาสที่แท้จริงสูงไปพร้อม ๆ กับพยายามทำให้ loss ของคลาสเป้าหมายต่ำที่สุดด้วย
นั่นคือ ถ้า $$x$$ เป็น input ที่มีคลาสที่ถูกต้องเป็น $$y$$ และเราต้องการก่อกวนให้ classifier ทำนายเป็นคลาส $$y'$$ เราทำได้โดยการแก้ปัญหา optimization ดังนี้

$$
\max_{\delta\in\Delta}(\ell(h_\theta(x+\delta),y) - \ell(h_\theta(x+\delta),y'))
$$

เนื่องจาก ใน softmax cross entropy loss ทั้งสองตัวนี้มีค่าของ $$\log\sum_{j=1}^ke^{h_\theta(x)_j}$$ ทั้งคู่ซึ่งจะถูกตัดกันไป เราจึงลดรูปปัญหาได้เป็น

$$
\max_{\delta\in\Delta}(h_\theta(x+\delta)_{y'} - h_\theta(x+\delta)_y)
$$

จากตัวอย่างรูปหมูเดิม เมื่อทดลองกำหนดเป้าหมายเป็นคลาสของเครื่องบิน และทำ projected gradient descent โดยใช้ learning rate เป็น 0.005 ปรากฏว่าหลังจากรันไป 100 รอบก็สามารถทำให้ ResNet50 ทำนายว่ารูปที่ถูกก่อกวนเป็นรูปเครื่องบินด้วยความน่าจะเป็น 0.968 โดยที่รูปที่ถูกก่อกวนเป็นดังนี้

<p align="center">
<img width="350" src="https://raw.githubusercontent.com/vacharapat/Adversarial-Machine-Learning/master/images/output_3.png">
</p>

และ noise ที่เราใช้ก่อกวนเป็นดังนี้ (เพิ่มความเข้มขึ้น 50 เท่า)

<p align="center">
<img width="350" src="https://raw.githubusercontent.com/vacharapat/Adversarial-Machine-Learning/master/images/output_4.png">
</p>

## References
1. [Z. Kolter, A. Madry, Adversarial Robustness - Theory and Practice](https://adversarial-ml-tutorial.org)

---
Prev: [การสร้าง adversarial example](https://vacharapat.github.io/Adversarial-Machine-Learning/docs/intro2)
Next: [Adversarial robustness](https://vacharapat.github.io/Adversarial-Machine-Learning/docs/intro4)
