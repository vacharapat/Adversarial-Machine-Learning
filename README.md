# Adversarial Machine Learning

### การเรียนรู้ของเครื่องแบบปฏิปักษ์: การก่อกวน และความทนทาน

เมื่อ machine learning เข้ามามีบทบาทในชีวิตจริงเป็นอย่างมากจนกระทั่งความผิดพลาดจากแบบจำลองอาจนำไปสู่ความเสียหายอย่างใหญ่หลวงหรือกระทั่งนำไปสู่การเสียชีวิต การสร้างแบบจำลองทาง machine learning จะคำนึงถึงแต่ความแม่นยำในการตัดสินใจไม่ได้อีกแล้ว เราจำเป็นต้องสนใจ _ความทนทาน_ (robustness) ของแบบจำลองด้วย ซึ่งหมายถึงความสามารถในการตัดสินใจที่มั่นคงเมื่อข้อมูลที่รับเข้ามามีการปนเปื้อนด้วย noise ที่เกิดจากสภาพแวดล้อมหรือแม้กระทั่ง noise ที่เกิดจาก _การก่อกวน_ (perturbation) จากผู้ประสงค์ร้าย

บทความชุดนี้บันทึกความรู้พื้นฐานทางทฤษฎีที่เกี่ยวข้องกับการก่อกวนแบบจำลองทาง machine learning ที่ได้รับความนิยมในปัจจุบันเช่น deep learning รวมถึงการเพิ่มความทนทานให้กับแบบจำลองเพื่อป้องกันการถูกก่อกวน
เนื้อหาส่วนใหญ่อ้างอิงจาก [Adversarial Robustness - Theory and Practice](https://adversarial-ml-tutorial.org) ของ Zico Kolter และ Aleksander Madry

### บทนำ

1. [แบบจำลอง deep learning พอสังเขป](https://vacharapat.github.io/Adversarial-Machine-Learning/docs/01)
1. [การสร้าง adversarial example](https://vacharapat.github.io/Adversarial-Machine-Learning/docs/02)
1. [การโจมตีแบบกำหนดเป้าหมาย](https://vacharapat.github.io/Adversarial-Machine-Learning/docs/03)
1. [Adversarial robustness](https://vacharapat.github.io/Adversarial-Machine-Learning/docs/04)
1. [การสร้าง robust classifier](https://vacharapat.github.io/Adversarial-Machine-Learning/docs/05)
1. [Linear binary classification](https://vacharapat.github.io/Adversarial-Machine-Learning/docs/06)
1. [Adversarial linear binary classification](https://vacharapat.github.io/Adversarial-Machine-Learning/docs/07)
