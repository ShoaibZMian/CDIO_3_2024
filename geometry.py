from time import sleep
from graphics import *
import math

#init window
win = GraphWin(width = 600, height = 600)
coordsWidth = 300
coordsHeight = 300
win.setCoords(0, 0, coordsWidth, coordsHeight)

def rotatePolygon(polygon: Polygon, degrees, center: Point) -> Polygon:
    rotatedVertices = []
    for point in polygon.points:
        newX = center.x + (point.x - center.x) * math.cos(math.radians(degrees)) - (point.y - center.x) * math.sin(math.radians(degrees))
        newY = center.x + (point.x - center.x) * math.sin(math.radians(degrees)) + (point.y - center.x) * math.cos(math.radians(degrees))
        rotatedVertices.append(Point(newX,newY))
    return Polygon(rotatedVertices)

def movePolygon(polygon: Polygon, dx, dy) -> Polygon:
    movedVertices = []
    for point in polygon.points:
        newX = point.x + dx
        newY = point.y + dy
        movedVertices.append(Point(newX,newY))
    return Polygon(movedVertices)

# draw axes
origin_offset = 150
xAxis = Line(Point(0, origin_offset), Point(coordsWidth, origin_offset)).draw(win)
yAxis = Line(Point(origin_offset, 0), Point(origin_offset, coordsHeight)).draw(win)


## define square paramters
center = Point(origin_offset + 15, origin_offset + 15)
side = 20
vertices = [Point(center.x-side/2, center.y-side/2), 
            Point(center.x+side/2, center.y-side/2),
            Point(center.x+side/2, center.y+side/2), 
            Point(center.x-side/2, center.y+side/2)]
square = Polygon(vertices)
square.setOutline('red')

# define newSquare
newSquare = Polygon(vertices)

# define pentagon
centerPentagon = Point(origin_offset - 50, origin_offset - 50)
pentagon = Polygon(Point(centerPentagon.x, centerPentagon.y + 20), 
                   Point(centerPentagon.x - 15, centerPentagon.y + 5),
                   Point(centerPentagon.x - 10, centerPentagon.y - 10),
                   Point(centerPentagon.x + 10, centerPentagon.y - 10),
                   Point(centerPentagon.x + 15, centerPentagon.y + 5))

# draw polygons
square.draw(win)
newSquare.draw(win)
pentagon.draw(win)

# animate polygons
nextMove = Point(2, 2)
while True:
    if win.checkKey():
        break
    else:
        sleep(0.1)
        square.undraw()
        square = rotatePolygon(square, -3, center)
        square.setOutline('red')
        square.draw(win)

        newSquare.undraw()
        newSquare = rotatePolygon(newSquare, 5, Point(origin_offset, origin_offset))
        newSquare.setOutline('green')
        newSquare.draw(win)

        pentagon.undraw()
        pentagon = rotatePolygon(pentagon, -10, centerPentagon)
        pentagon.setOutline('purple')
        pentagon.draw(win)