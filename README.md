# Adversarial Machine Learning

### การเรียนรู้ของเครื่องแบบปฏิปักษ์: การก่อกวน และความทนทาน

เมื่อ machine learning เข้ามามีบทบาทในชีวิตจริงเป็นอย่างมากจนกระทั่งความผิดพลาดจากแบบจำลองอาจนำไปสู่ความเสียหายอย่างใหญ่หลวงหรือกระทั่งนำไปสู่การเสียชีวิต การสร้างแบบจำลองทาง machine learning จะคำนึงถึงแต่ความแม่นยำในการตัดสินใจไม่ได้อีกแล้ว เราจำเป็นต้องสนใจ _ความทนทาน_ (robustness) ของแบบจำลองด้วย ซึ่งหมายถึงความสามารถในการตัดสินใจที่มั่นคงเมื่อข้อมูลที่รับเข้ามามีการปนเปื้อนด้วย noise ที่เกิดจากสภาพแวดล้อมหรือแม้กระทั่ง noise ที่เกิดจาก _การก่อกวน_ (perturbation) จากผู้ประสงค์ร้าย

บทความชุดนี้บันทึกความรู้พื้นฐานทางทฤษฎีที่เกี่ยวข้องกับการก่อกวนบนแบบจำลองทาง machine learning ที่ได้รับความนิยมในปัจจุบันเช่น deep learning, การเพิ่มความทนทานให้กับแบบจำลองเพื่อป้องกันการถูกก่อกวน รวมถึงความรู้ทางทฤษฎีในมุมต่าง ๆ ที่เกี่ยวข้องกับความทนทานของแบบจำลองเหล่านี้
โดยเนื้อหาในช่วงแรกอ้างอิงจาก [Adversarial Robustness - Theory and Practice](https://adversarial-ml-tutorial.org) ของ Zico Kolter และ Aleksander Madry

### บทนำ
1. [แบบจำลอง deep learning พอสังเขป](https://vacharapat.github.io/Adversarial-Machine-Learning/docs/01)
1. [การสร้าง adversarial example](https://vacharapat.github.io/Adversarial-Machine-Learning/docs/02)
1. [การโจมตีแบบกำหนดเป้าหมาย](https://vacharapat.github.io/Adversarial-Machine-Learning/docs/03)
1. [Adversarial robustness](https://vacharapat.github.io/Adversarial-Machine-Learning/docs/04)
1. [การสร้าง robust classifier](https://vacharapat.github.io/Adversarial-Machine-Learning/docs/05)

### การโจมตีและการป้องกัน
1. [แบบจำลอง Linear binary classification](https://vacharapat.github.io/Adversarial-Machine-Learning/docs/06)
1. [Adversarial linear binary classification](https://vacharapat.github.io/Adversarial-Machine-Learning/docs/07)
1. [Fast Gradient Sign Method](https://vacharapat.github.io/Adversarial-Machine-Learning/docs/08)
1. [Projected Gradient Descent](https://vacharapat.github.io/Adversarial-Machine-Learning/docs/09)
1. [Adversarial training](https://vacharapat.github.io/Adversarial-Machine-Learning/docs/10)

### Robustness Certification
1. [การสร้าง robustness certificate](https://vacharapat.github.io/Adversarial-Machine-Learning/docs/11)
1. [Linear programming relaxation](https://vacharapat.github.io/Adversarial-Machine-Learning/docs/12)
1. [Quadratic programming](https://vacharapat.github.io/Adversarial-Machine-Learning/docs/13)
1. [Semidefinite programming relaxation](https://vacharapat.github.io/Adversarial-Machine-Learning/docs/14)

### ทฤษฎีการเรียนรู้เชิงคำนวณ
1. [การเรียนรู้แบบ Probably approximately correct](https://vacharapat.github.io/Adversarial-Machine-Learning/docs/15)
1. [ปัญหาการเรียนรู้รูปสี่เหลี่ยมขนานแกน](https://vacharapat.github.io/Adversarial-Machine-Learning/docs/16)
1. [การเรียนรู้เมื่อ hypothesis space มีขนาดจำกัด](https://vacharapat.github.io/Adversarial-Machine-Learning/docs/17)
1. [Agnostic PAC learning](https://vacharapat.github.io/Adversarial-Machine-Learning/docs/18)

### VC Dimension
1. [Growth function และ VC dimension](https://vacharapat.github.io/Adversarial-Machine-Learning/docs/19)
1. [Generalization bound](https://vacharapat.github.io/Adversarial-Machine-Learning/docs/20)
1. [Sample complexity](https://vacharapat.github.io/Adversarial-Machine-Learning/docs/21)
1. Lower bound

### Adversarially Robust Generalization
1. Adversarial agnostic PAC learning
1. Adversarial VC dimension
1. Sample complexity สำหรับ Gaussian model
1. Sample complexity สำหรับ Bernoulli model

