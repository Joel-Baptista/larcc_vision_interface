from moviepy.editor import VideoFileClip, concatenate_videoclips
import cv2

save_path = "/home/joelbaptista/Desktop/"
clip = VideoFileClip("/home/joelbaptista/Desktop/demonstration.avi")

clip1 = clip.subclip(0, 63.8)
clip3 = clip.subclip(93, 110)
clip2 = VideoFileClip("/home/joelbaptista/Desktop/subclip.avi")


final = concatenate_videoclips([clip1, clip2, clip3])
#writing the video into a file / saving the combined video
final.write_videofile("/home/joelbaptista/Desktop/new_demonstration.avi", codec="libx264")

# writer = cv2.VideoWriter(save_path + 'subclip.avi',
#                                       cv2.VideoWriter_fourcc('I', '4', '2', '0'),
#                                       30,
#                                       (640, 480))
#
# vidcap = cv2.VideoCapture("/home/joelbaptista/Desktop/demonstration.avi")
# success, image = vidcap.read()
#
# count = 0
# while success:
#     if (5*count) % 2 != 0 and 1274 <= count <= 1859:
#         writer.write(image)
#
#     # if count<10:
#     #     id = f'00{count}'
#     # elif count < 100:
#     #     id = f'0{count}'
#     # else:
#     #     id = count
#     # cv2.imwrite(f"/home/joelbaptista/Desktop/new_frames/frame{id}.jpg", image)    # save frame as JPEG file
#     success, image = vidcap.read()
#     count += 1