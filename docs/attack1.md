{% include lib/mathjax.html %}
# แบบจำลอง Linear binary classification

หลังจากที่เราได้รู้จักหลักในการสร้าง adversarial example และการเทรน robust model กันแล้ว เราจะลองนำหลักดังกล่าวมาวิเคราะห์บนแบบจำลอง classification ที่ง่ายที่สุดก่อน นั่นคือแบบจำลองเชิงเส้น (linear model) ซึ่งจะใช้ฟังก์ชันเชิงเส้น (linear function) ในการจำแนกข้อมูล หรือกล่าวได้ว่าเราสามารถแทนแบบจำลองได้ด้วยฟังก์ชัน $$h_\theta:\mathbb{R}^n\rightarrow\mathbb{R}^k$$ ดังนี้

$$
h_\theta(x)=Wx+b
$$

เมื่อ $$\theta=\{W\in\mathbb{R}^{k\times n}, b\in\mathbb{R}^k\}$$

## Logistic regression

เพื่อความง่าย เราจะเริ่มพิจารณากรณีที่ข้อมูลมี 2 กลุ่มเท่านั้น ($$k=2$$) นั่นคือ เราสนใจการสร้างฟังก์ชันเชิงเส้นสำหรับปัญหา binary classification

สำหรับ loss function ในกรณีที่เราต้องการจำแนกข้อมูลเป็นสองกลุ่ม (กลุ่มที่ 1 และกลุ่มที่ 2) หากเราใช้แบบจำลองที่คืนค่า $$h_\theta(x)$$ เป็นเวกเตอร์สองมิติ และใช้ฟังก์ชัน softmax ในการคำนวณความน่าจะเป็นที่ตัวอย่างข้อมูล $$x$$ ใด ๆ จะอยู่ในแต่ละกลุ่ม จะได้ว่าความน่าจะเป็นที่ $$x$$ จะอยู่ในกลุ่มที่ 1 มีค่าเป็น

$$
\frac{e^{h_\theta(x)_1}}{e^{h_\theta(x)_1}+e^{h_\theta(x)_2}}
= \frac{1}{1+e^{h_\theta(x)_2-h_\theta(x)_1}}
$$

และความน่าจะเป็นที่ $$x$$ จะอยู่ในกลุ่มที่ 2 มีค่าเป็น

$$
\frac{e^{h_\theta(x)_2}}{e^{h_\theta(x)_1}+e^{h_\theta(x)_2}}
= \frac{1}{1+e^{-(h_\theta(x)_2-h_\theta(x)_1)}}
$$

เมื่อ $$h_\theta(x)_1=w_1^Tx+b_1$$ และ $$h_\theta(x)_2=w_2^Tx+b_2$$

ดังนั้นจะเห็นว่า เราสามารถใช้ฟังก์ชันเชิงเส้น $$h'_\theta$$ ที่คืนค่าเป็นจำนวนจริง

$$
h'_\theta(x)=h_\theta(x)_1-h_\theta(x)_2=(w_1-w_2)^Tx+(b_1-b_2)
$$

เป็นตัวแทนการตัดสินใจของ $$h_\theta$$ ได้เช่นกัน โดยถ้าเราให้ข้อมูลกลุ่มแรกมี label เป็น $$+1$$ และกลุ่มที่สองมี label เป็น $$-1$$ จะได้ว่า
ความน่าจะเป็นที่ตัวอย่างข้อมูล $$x$$ จะมี label เป็น $$y$$ คือ

$$
\Pr[y|x]=\frac{1}{1+e^{-y\cdot h'_\theta(x)}}
$$

ดังนั้น สำหรับปัญหา linear binary classification เราสามารถนิยามแบบจำลองเชิงเส้นได้ด้วยฟังก์ชัน $$h_\theta:\mathbb{R}^n\rightarrow\mathbb{R}$$ ดังนี้

$$
h_\theta(x)=w^Tx+b
$$

เมื่อ $$\theta=\{w\in\mathbb{R}^n,b\in\mathbb{R}\}$$ โดยมี class label เป็น $$y\in\{+1,-1\}$$ และ loss function คำนวณได้จาก

$$
\begin{split}
\ell(h_\theta(x),y) &=-\log\Pr[y|x]\\
&=-\log\frac{1}{1+e^{-y\cdot h_\theta(x)}}\\
&= \log(1+e^{-y\cdot h_\theta(x)})
\end{split}
$$

เราเรียกแบบจำลองนี้ว่า logistic regression และเรียก loss function นี้ว่า logistic loss เพื่อความสะดวกเรานิยามให้ $$L(z)=\log(1+e^{-z})$$ เราจะเขียนแทน logistic loss ได้ด้วย $$L(y\cdot h_\theta(x))$$

## References
1. [Z. Kolter, A. Madry. Adversarial Robustness - Theory and Practice](https://adversarial-ml-tutorial.org)

---
Prev: [การสร้าง robust classifier](https://vacharapat.github.io/Adversarial-Machine-Learning/docs/intro5)

Next: [Adversarial linear binary classification](https://vacharapat.github.io/Adversarial-Machine-Learning/docs/attack2)
