from graphics import *
import math

#init window
win = GraphWin(width = 600, height = 600)
win.setCoords(0, 0, 100, 100)

def rotateRectangle(rectangle: Polygon, degrees, centerX, centerY) -> Polygon:
    if len(rectangle.points) != 4:
        raise Exception(f"Provided polygon must be a rectangle, this one has {len(rectangle.points)} vertices instead of 4")
    rotatedVertices = []
    for point in rectangle.points:
        x = centerX + point.x * math.cos(math.radians(degrees)) - point.y * math.sin(math.radians(degrees))
        y = centerY + point.y * math.cos(math.radians(degrees)) + point.x * math.sin(math.radians(degrees))
        rotatedVertices.append(Point(x,y))

    return Polygon(rotatedVertices)

# draw axes
xAxis = Line(Point(0, 50), Point(100, 50)).draw(win)
yAxis = Line(Point(50, 0), Point(50, 100)).draw(win)


# draw test square
center = 55
side = 20

vertices = [Point(center-side/2, center-side/2), Point(center+side/2, center-side/2), Point(center+side/2, center+side/2), Point(center-side/2, center+side/2)]
square = Polygon(vertices)
square.setOutline('red')
square.draw(win)

newSquare = rotateRectangle(square, 45, 0, 0)
newSquare.setOutline('green')
newSquare.draw(win)

# wait for input before closing
win.getKey()


# swap boxes
#switchSquare = 0
#while True:
#    if win.getKey():
#        if switchSquare == 0:
#            mySquare.undraw()
#            newSquare.draw(win)
#            switchSquare = 1
#        else:
#            newSquare.undraw()
#            mySquare.draw(win)
#            switchSquare = 0