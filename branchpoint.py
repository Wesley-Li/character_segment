import cv2
import numpy
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
args = vars(ap.parse_args())

frame = cv2.imread(args['image'])

if frame is None:
    print('Error loading image')
    exit()

colour_frame = frame

frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

ret, frame = cv2.threshold(frame, 127, 255, cv2.THRESH_BINARY)

cv2.imshow("Binary", frame)
cv2.waitKey(0)

rows = frame.shape[0]
cols = frame.shape[1]

branch_locations = []

# start with second column
for i in range(1, cols):
    lit = False
    begin_black_regions = []
    end_black_regions = []

    # start with first row
    if 255 == frame[0, i]:
        lit = True
    else:
        lit = False
        begin_black_regions.append(0)

    # start with second row
    for j in range(1, rows - 1):
        if 255 == frame[j, i] and not lit:
            lit = True
            end_black_regions.append(j - 1)
        elif frame[j, i] == 0 and lit:
            lit = False
            begin_black_regions.append(j)

    # end with last row
    if 0 == frame[rows - 1, i] and not lit:
        end_black_regions.append(rows - 1)
    elif 0 == frame[rows - 1, i] and lit:
        begin_black_regions.append(rows - 1)
        end_black_regions.append(rows - 1)
    elif 255 == frame[rows - 1, i] and not lit:
        end_black_regions.append(rows - 2)

    for k in range(0, len(begin_black_regions)):
        found_branch = True

        for l in range(begin_black_regions[k], end_black_regions[k] + 1):
            if 0 == frame[l, i - 1]:
                found_branch = False
                break

        if found_branch:
            branch_locations.append(complex(i - 1, begin_black_regions[k]))

for i in range(0, len(branch_locations)):
    cv2.circle(colour_frame, (int(branch_locations[i].real), int(branch_locations[i].imag)), 2, (255, 127, 0), 2)

cv2.imshow("Frame", colour_frame)

cv2.waitKey(0)
