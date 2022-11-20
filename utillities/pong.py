# Import required library
import turtle
from pynput.keyboard import Key, Controller
import random
import time

# source: https://www.geeksforgeeks.org/create-pong-game-using-python-turtle/

# run faster
# https://stackoverflow.com/questions/59119036/how-do-i-make-a-turtle-run-faster

# simulate key presses
# https://www.youtube.com/watch?v=nQTWe5PYDwM&t=107s&ab_channel=Pydemy


keyboard = Controller()

# Create screen
sc = turtle.Screen()
sc.title("Pong game")
sc.bgcolor("white")
sc.setup(width=1000, height=600)

# Midfield
midfield = turtle.Turtle()
midfield.speed(1)
midfield.shape("square")
midfield.color("black")
midfield.shapesize(stretch_wid=25, stretch_len=0.2)
midfield.penup()
midfield.goto(0, 0)

# Left Penalty
midfield = turtle.Turtle()
midfield.speed(1)
midfield.shape("square")
midfield.color("black")
midfield.shapesize(stretch_wid=25, stretch_len=0.2)
midfield.penup()
midfield.goto(-250, 0)

# Right Penalty
midfield = turtle.Turtle()
midfield.speed(1)
midfield.shape("square")
midfield.color("black")
midfield.shapesize(stretch_wid=25, stretch_len=0.2)
midfield.penup()
midfield.goto(250, 0)


# Left paddle
left_pad = turtle.Turtle()
left_pad.speed(0)
left_pad.shape("square")
left_pad.color("blue")
left_pad.shapesize(stretch_wid=6, stretch_len=2)
left_pad.penup()
left_pad.goto(-400, 0)


# Right paddle
right_pad = turtle.Turtle()
right_pad.speed(0)
right_pad.shape("square")
right_pad.color("red")
right_pad.shapesize(stretch_wid=6, stretch_len=2)
right_pad.penup()
right_pad.goto(400, 0)


# Ball of circle shape
hit_ball = turtle.Turtle()
hit_ball.speed(0)
hit_ball.shape("circle")
hit_ball.color("black")
hit_ball.penup()
hit_ball.goto(0, 0)
hit_ball.dx = 5
hit_ball.dy = -5


# Initialize the score
left_player = 0
right_player = 0


# Displays the score
sketch = turtle.Turtle()
sketch.speed(1)
sketch.color("blue")
sketch.penup()
sketch.hideturtle()
sketch.goto(0, 260)
sketch.write(
    "Left_player : 0    Right_player: 0", align="center", font=("Courier", 24, "normal")
)


# Functions to move paddle vertically
def paddleaup():
    y = left_pad.ycor()
    y += 20
    left_pad.sety(y)


def paddleadown():
    y = left_pad.ycor()
    y -= 20
    left_pad.sety(y)


def paddlebup():
    y = right_pad.ycor()
    y += 20
    right_pad.sety(y)


def paddlebdown():
    y = right_pad.ycor()
    y -= 20
    right_pad.sety(y)


def paddlebleft():
    x = right_pad.xcor()
    x -= 20
    right_pad.setx(x)


def paddlebright():
    x = right_pad.xcor()
    x += 20
    right_pad.setx(x)

# test method
def test():
    print("hit right paddle")

# Keyboard bindings
sc.listen()
sc.onkeypress(paddleaup, "e")
sc.onkeypress(paddleadown, "x")

sc.onkeypress(paddlebup, "Up")
sc.onkeypress(paddlebdown, "Down")
sc.onkeypress(paddlebleft, "Left")
sc.onkeypress(paddlebright, "Right")

# test listener
sc.onkeypress(test, "i")


while True:
    sc.update()

    hit_ball.setx(hit_ball.xcor() + hit_ball.dx)
    hit_ball.sety(hit_ball.ycor() + hit_ball.dy)

    # Checking borders
    if hit_ball.ycor() > 280:
        hit_ball.sety(280)
        hit_ball.dy *= -1

    if hit_ball.ycor() < -280:
        hit_ball.sety(-280)
        hit_ball.dy *= -1

    if hit_ball.xcor() > 500:
        hit_ball.goto(0, 0)
        left_player += 1
        sketch.clear()
        sketch.write(
            "Left_player : {}    Right_player: {}".format(left_player, right_player),
            align="center",
            font=("Courier", 24, "normal"),
        )

    if hit_ball.xcor() < -500:
        hit_ball.goto(0, 0)
        right_player += 1
        sketch.clear()
        sketch.write(
            "Left_player : {}    Right_player: {}".format(left_player, right_player),
            align="center",
            font=("Courier", 24, "normal"),
        )

    # Paddle ball collision
    rp_collision = right_pad.xcor() - 30
    if (hit_ball.xcor() == rp_collision) and (
        hit_ball.ycor() < right_pad.ycor() + 40
        and hit_ball.ycor() > right_pad.ycor() - 40
    ):
        
        # test - simulate keyboard presses
        keyboard.press('i')
        hit_ball.setx(rp_collision)
        hit_ball.dx *= -1

    lp_collision = left_pad.xcor() + 30
    if (hit_ball.xcor() == lp_collision) and (
        hit_ball.ycor() < left_pad.ycor() + 40
        and hit_ball.ycor() > left_pad.ycor() - 40
    ):
        hit_ball.setx(lp_collision)
        hit_ball.dx *= -1
