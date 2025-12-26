from __future__ import annotations
# 어떤 것을 처리하기 위한 부품을 만들어버리자.
# 여러 부품을 만들 때 틀(클래스)을 먼저만들어두고
# 부품이 필요하면 틀을 찍어서 사용할 대상(부품)을 만든다.
# Car 틀을 먼저 만든다.
# --> 내차가 필요하면 Car틀을 이용해서 차를 만들고
# --> 내차의 특징을 상세하게 넣어줌.
# 만들 대상(부품, object, 객체)
# --> 대상(object, 객체)로 만들어서 코딩하는 방식
# --> 객체 지향형 프로그래밍(Object Oriented Programming, OOP)
class Car:  #차에 대한 틀을 만들자 --> 차에 일반적인 특징으로 만들자!
    # 특징(속성)
    # price : int = 1000
    # color : str = "검정색"

    # my_car = Car()자동 호출됨.
    def __init__(self, price, color):
        self.price = price
        self.color = color

    # print(my_car) 자동 호출됨.
    def __str__(self):
        return str(self.price) + " "+ self.color

    # 특징(동작)
    def run(self):
        print("달리다.")

    def speed(self):
        print("스피드를 올리다.")

    @staticmethod ##annotations(표시)
    def start():
        print("만든 회사이름은 현대자동차이다.")

if __name__ == '__main__':
    # my_car만들때 값들 자동으로 초기화하자.
    my_car = Car(3000, "red")
    print(my_car)
    my_car.speed()

    Car.start()

    # my_car = Car()
    # #기본 설정되어있는 값
    # print(my_car.price, my_car.color)
    # my_car.price = 3000
    # my_car.color = "red"
    # print(my_car.price, my_car.color)
    #
    # your_car = Car()
    # print(your_car.price, your_car.color)