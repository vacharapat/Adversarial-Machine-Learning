{% include lib/mathjax.html %}
# Adversarial linear binary classification

ในหัวข้อนี้เราจะมาวิเคราะห์การสร้าง adversarial example สำหรับแบบจำลอง logistic regression และการเทรน logistic regression classifier ให้มีความทนทานต่อการก่อกวน

## การสร้าง adversarial example

เมื่อเราพิจารณาการสร้าง adversarial example หรือการก่อกวนแบบจำลอง logistic regression นี้ด้วยปัญหา maximization ในปัญหา min-max robust optimization เราจะได้ว่าปัญหาของเราคือ

$$
\max_{\delta\in\Delta}\ell(h_\theta(x+\delta),y)=\max_{\delta\in\Delta}L(y\cdot (w^T(x+\delta)+b))
$$

สิ่งที่น่าสนใจสำหรับกรณีของ logistic regression นี้คือเราสามารถแก้ปัญหา maximization นี้ได้ เนื่องจากฟังก์ชัน $$L$$ ที่เราใช้นั้นมีลักษณะเป็น monotonically decreasing กล่าวคือค่าของ $$L(z)$$ จะมีค่าลดลงเสมอเมื่อ $$z$$ มีค่าเพิ่มขึ้น หากเราลองพล็อตกราฟระหว่าง $$z$$ และ $$L(z)$$ จะได้รูปดังตัวอย่าง

<p align="center">
<img width="350" src="https://raw.githubusercontent.com/vacharapat/Adversarial-Machine-Learning/master/images/output_5.png">
</p>

จากคุณสมบัติของฟังก์ชัน $$L$$ นี้ จะเห็นว่าหากเราต้องการให้ $$L$$ คืนผลลัพธ์ที่มีค่ามากที่สุด เราทำได้โดยส่งค่าที่น้อยที่สุดเข้าไปให้ $$L$$ นั่นเอง นั่นคือ

$$
\begin{split}
\max_{\delta\in\Delta}L(y\cdot(w^T(x+\delta)+b)) &=L(\min_{\delta\in\Delta}y\cdot(w^T(x+\delta)+b))\\
&=L(y\cdot(w^Tx+b)+\min_{\delta\in\Delta}y\cdot w^T\delta)
\end{split}
$$

จะได้ว่าปัญหาของเราลดรูปมาเป็น

$$
\min_{\delta\in\Delta}y\cdot w^T\delta
$$

สมมติให้ $$\Delta$$ เป็น $$\ell_\infty$$-norm $$\Delta=\{\delta:\|\delta\|_\infty\leq\epsilon\}$$
หากเราพิจารณากรณีที่ $$y=+1$$ จะเห็นว่า เราสามารถหาคำตอบของปัญหา minimization นี้ได้โดยกำหนดให้ $$\delta_i=-\epsilon$$
เมื่อ $$w_i\geq 0$$ และให้ $$\delta_i=\epsilon$$ เมื่อ $$w_i<0$$ ในกรณีที่ $$y=-1$$ เราก็เพียงสลับเครื่องหมายของ $$\delta_i$$ ตามเงื่อนไขเหล่านี้ เราก็จะสามารถหา $$\delta^*$$ ที่เป็นคำตอบที่ดีที่สุดได้ เราอาจเขียนคำตอบนี้ได้เป็น

$$
\delta^*=-y\epsilon\cdot\text{sign}(w)
$$

ซึ่งเมื่อเราแทนค่า $$\delta^*$$ เข้าไปในฟังก์ชันที่ต้องการค่าน้อยที่สุด จะได้ว่า

$$
\begin{split}
\min_{\|\delta\|_\infty\leq\epsilon}y\cdot w^T\delta &=y\cdot w^T\delta^*\\
&=y\cdot\sum_{i=1}^n-y\epsilon\cdot\text{sign}(w_i)w_i\\
&=-y^2\epsilon\sum_{i=1}^n|w_i|\\
&=-\epsilon\|w\|_1
\end{split}
$$

ดังนั้นเราสามารถแก้ปัญหา maximization ในการสร้าง adversarial example ได้ดังนี้

$$
\begin{split}
\max_{\|\delta\|_\infty\leq\epsilon}\ell(h_\theta(x+\delta),y) &=\max_{\|\delta\|_\infty\leq\epsilon}L(y\cdot(w^T(x+\delta)+b))\\
&=L(y\cdot (w^Tx+b)-\epsilon\|w\|_1)
\end{split}
$$

## การเทรน robust classifier

จากปัญหา min-max robust optimization เมื่อเราสามารถแก้ปัญหา maximization ข้างในได้ การหาพารามิเตอร์ที่เหมาะสมสำหรับแบบจำลองที่มีความทนทานก็ทำได้โดยการแก้ปัญหา minimization ต่อไปนี้

$$
\min_{w,b}\frac{1}{|D_\text{train}|}\sum_{(x,y)\in D_\text{train}}L(y\cdot (w^Tx+b)-\epsilon\|w\|_1)
$$

เมื่อ $$D_\text{train}$$ เป็นเซตของ training data ซึ่งปัญหานี้มีคุณสมบัติ convex บนตัวแปร $$w$$ และ $$b$$ ดังนั้นอัลกอริทึมในกลุ่ม gradient descent เช่น SGD ก็สามารถนำไปสู่คำตอบที่ดีที่สุดได้

## References
1. [Z. Kolter, A. Madry. Adversarial Robustness - Theory and Practice](https://adversarial-ml-tutorial.org)

---
Prev: [แบบจำลอง Linear binary classification](https://vacharapat.github.io/Adversarial-Machine-Learning/docs/attack1)

Next: [Fast Gradient Sign Method](https://vacharapat.github.io/Adversarial-Machine-Learning/docs/attack3)
