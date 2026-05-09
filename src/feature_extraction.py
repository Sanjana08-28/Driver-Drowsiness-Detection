import math


def euclidean_distance(point1, point2):

    return math.sqrt(
        (point1[0] - point2[0]) ** 2 +
        (point1[1] - point2[1]) ** 2
    )


def calculate_ear(landmarks, eye_indices, w, h):

    points = []

    for idx in eye_indices:

        x = int(landmarks[idx].x * w)
        y = int(landmarks[idx].y * h)

        points.append((x, y))

    p1, p2, p3, p4, p5, p6 = points

    vertical1 = euclidean_distance(p2, p6)
    vertical2 = euclidean_distance(p3, p5)

    horizontal = euclidean_distance(p1, p4)

    ear = (vertical1 + vertical2) / (2.0 * horizontal)

    return ear


def calculate_mar(landmarks, mouth_indices, w, h):

    points = []

    for idx in mouth_indices:

        x = int(landmarks[idx].x * w)
        y = int(landmarks[idx].y * h)

        points.append((x, y))

    top = points[0]
    bottom = points[1]

    left = points[2]
    right = points[3]

    vertical = euclidean_distance(top, bottom)

    horizontal = euclidean_distance(left, right)

    mar = vertical / horizontal

    return mar