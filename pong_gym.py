import turtle
import numpy as np

# original game:
# https://www.geeksforgeeks.org/create-pong-game-using-python-turtle/

# run faster
# https://stackoverflow.com/questions/59119036/how-do-i-make-a-turtle-run-faster

# simulate key presses
# https://www.youtube.com/watch?v=nQTWe5PYDwM&t=107s&ab_channel=Pydemy


class Pong:
    def __init__(self):
        # Init score
        self.left_player = 0
        self.right_player = 0
        # Left pad
        self.left_pad = turtle.Turtle()
        # Right pad
        self.right_pad = turtle.Turtle()
        # Ball of circle shape
        self.hit_ball = turtle.Turtle()

        # Create screen
        self.sc = turtle.Screen()
        self.sc.title("Pong Game")
        self.sc.bgcolor("white")
        self.sc.setup(width=1000, height=600)

        # Keyboard bindings
        self.sc.listen()
        self.sc.onkeypress(self.pad_a_up, "e")
        self.sc.onkeypress(self.pad_a_down, "x")

        self.sc.onkeypress(self.pad_b_up, "Up")
        self.sc.onkeypress(self.pad_b_down, "Down")
        self.sc.onkeypress(self.pad_b_left, "Left")
        self.sc.onkeypress(self.pad_b_right, "Right")

        # Displays the score
        self.sketch = turtle.Turtle()
        self.sketch.speed(1)
        self.sketch.color("blue")
        self.sketch.penup()
        self.sketch.hideturtle()
        self.sketch.goto(0, 260)
        self.sketch.write(
            "Left_player : 0    Right_player: 0",
            align="center",
            font=("Courier", 24, "normal"),
        )

        # Left Penalty
        self.lf_penalty = turtle.Turtle()
        self.lf_penalty.speed(1)
        self.lf_penalty.shape("square")
        self.lf_penalty.color("black")
        self.lf_penalty.shapesize(stretch_wid=25, stretch_len=0.2)
        self.lf_penalty.penup()
        self.lf_penalty.goto(-250, 0)

        # Right Penalty
        self.rh_penalty = turtle.Turtle()
        self.rh_penalty.speed(1)
        self.rh_penalty.shape("square")
        self.rh_penalty.color("black")
        self.rh_penalty.shapesize(stretch_wid=25, stretch_len=0.2)
        self.rh_penalty.penup()
        self.rh_penalty.goto(250, 0)

        self.dy = 5
        self.dx = 5
        self.label = 4
        self.silent = False
        self.hit_ball.dx = 5

    def reset(self):
        # reset scores
        self.left_player = 0
        self.right_player = 0
        self.update_score_board()

        # left pad
        self.left_pad.speed(0)
        self.left_pad.shape("square")
        self.left_pad.color("blue")
        self.left_pad.shapesize(stretch_wid=6, stretch_len=2)
        self.left_pad.penup()
        self.left_pad.goto(-400, 0)

        # right pad
        self.right_pad.speed(0)
        self.right_pad.shape("square")
        self.right_pad.color("red")
        self.right_pad.shapesize(stretch_wid=6, stretch_len=2)
        self.right_pad.penup()
        self.right_pad.goto(400, 0)

        # ball of circle shape
        self.hit_ball.speed(0)
        self.hit_ball.shape("circle")
        self.hit_ball.color("black")
        self.hit_ball.penup()
        self.hit_ball.goto(0, 0)

        # self.hit_ball.dx = 5
        rng = np.random.default_rng()
        self.hit_ball.dy = rng.uniform(low=-5.0, high=5.0)

        # update screen
        self.render()

        return [
            self.left_pad.xcor(),
            self.left_pad.ycor(),
            self.right_pad.xcor(),
            self.right_pad.ycor(),
            self.hit_ball.xcor(),
            self.hit_ball.ycor(),
            self.hit_ball.dx,
            self.hit_ball.dy,
        ]

    # functions to move pad vertically
    def pad_a_up(self):
        y = self.left_pad.ycor()
        y += self.dy
        self.left_pad.sety(y)

    def pad_a_down(self):
        y = self.left_pad.ycor()
        y -= self.dy
        self.left_pad.sety(y)

    def pad_b_up(self):
        self.label = 0
        y = self.right_pad.ycor()
        y += self.dy
        self.right_pad.sety(y)

    def pad_b_down(self):
        self.label = 1
        y = self.right_pad.ycor()
        y -= self.dy
        self.right_pad.sety(y)

    def pad_b_left(self):
        self.label = 2
        x = self.right_pad.xcor()
        x -= self.dx
        self.right_pad.setx(x)

    def pad_b_right(self):
        self.label = 3
        x = self.right_pad.xcor()
        x += self.dx
        self.right_pad.setx(x)

    def update_score_board(self):
        self.sketch.clear()
        self.sketch.write(
            "Left_player : {}    Right_player: {}".format(
                self.left_player, self.right_player
            ),
            align="center",
            font=("Courier", 24, "normal"),
        )

    def is_paddle_within_bounds(self):

        if (
            self.right_pad.ycor() < 280
            and self.right_pad.ycor() > -280
            and self.right_pad.xcor() < 500
            # and self.right_pad.xcor() > -250
        ):
            return True
        else:
            return False

    def step(self, action, timestep):
        # move paddle given action
        
        if self.is_paddle_within_bounds():
            if action >= 0 and action < 0.5:
                self.pad_b_up()
            elif action >= 0.5 and action < 1.5:
                self.pad_b_down()
            elif action >= 1.5 and action < 2.5:
                self.pad_b_left()
            elif action >= 2.5 and action < 3.5:
                self.pad_b_right()
            # else do nothing

        # step rewward
        reward = 0

        # checking borders
        if self.hit_ball.ycor() > 280:
            self.hit_ball.sety(280)
            self.hit_ball.dy *= -1

        if self.hit_ball.ycor() < -280:
            self.hit_ball.sety(-280)
            self.hit_ball.dy *= -1

        if self.hit_ball.xcor() > 500:
            self.hit_ball.goto(0, 0)
            self.left_player += 1
            self.hit_ball.dx *= -1
            self.update_score_board()
            # reward for losing
            # reward = reward - 10

        if self.hit_ball.xcor() < -500:
            self.hit_ball.goto(0, 0)
            self.right_player += 1
            self.hit_ball.dx *= -1
            self.update_score_board()
            # reward for winning
            reward = reward + 10
            self.silent = False

        # pad ball collision
        rp_xcollision = self.right_pad.xcor() - 30
        if (self.hit_ball.xcor() == rp_xcollision) and (
            self.hit_ball.ycor() < self.right_pad.ycor() + 40
            and self.hit_ball.ycor() > self.right_pad.ycor() - 40
        ):
            # cheat - help to not double hit
            self.right_pad.setx(rp_xcollision + 100)
            self.hit_ball.setx(rp_xcollision)
            self.hit_ball.dx *= -1
            # reward for hitting the ball
            reward = reward + 5

        lp_xcollision = self.left_pad.xcor() + 30
        if self.hit_ball.xcor() == lp_xcollision:
            # move pad to y-ball
            y_ball = self.hit_ball.ycor()
            y_lpad = self.left_pad.ycor()
            y_delta = y_ball - y_lpad
            if y_delta >= 0:
                self.left_pad.sety(y_lpad + y_delta)
            else:
                self.left_pad.sety(y_lpad + y_delta)

            # hit balls
            self.hit_ball.setx(lp_xcollision)
            self.hit_ball.dx *= -1

        # hit ball
        self.hit_ball.setx(self.hit_ball.xcor() + self.hit_ball.dx)
        self.hit_ball.sety(self.hit_ball.ycor() + self.hit_ball.dy)

        # runs faster
        if self.silent == False:
            self.sc.update()

        state = [
            self.left_pad.xcor(),
            self.left_pad.ycor(),
            self.right_pad.xcor(),
            self.right_pad.ycor(),
            self.hit_ball.xcor(),
            self.hit_ball.ycor(),
            self.hit_ball.dx,
            self.hit_ball.dy,
        ]

        done = False
        if self.left_player == 100 or self.right_player == 100:
            done = True
            self.reset()

        return state, reward, done, self.label

    def render(self):
        self.sc.update()

    def set_silent(self, silent):
        if silent == True:
            self.sc.tracer(0)
        else:
            self.sc.tracer(1)
        self.silent = silent