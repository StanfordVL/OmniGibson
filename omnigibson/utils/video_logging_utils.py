import datetime
import os

import cv2
from matplotlib import font_manager
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch as th

import omnigibson as og


class VideoLogger:
    def __init__(self, args, env=None):
        self.vid_downscale_factor = args.vid_downscale_factor
        assert isinstance(args.vid_speedup, int)
        self.vid_speedup = args.vid_speedup
        self.env = env
        # ^ only needed if using save_im_text(...) instead of save_obs(...)
        now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.out_dir = os.path.join(args.out_dir, now)
        os.makedirs(self.out_dir, exist_ok=False)
        self.clear_ims()

        # Text settings
        self.text_font_size = 36
        self.line_spacing = 1.2
        self.num_frames_to_show_text = 45 * args.vid_speedup  # num frames to keep text on for the video
        self.text_to_num_frames_remaining_map = {}
        font = font_manager.FontProperties(family='Ubuntu', style="italic")
        italics = ImageFont.truetype(font_manager.findfont(font), self.text_font_size)
        font = font_manager.FontProperties(family='Ubuntu')
        non_italics = ImageFont.truetype(font_manager.findfont(font), self.text_font_size)
        self.fonts = dict(
            italics=italics,
            non_italics=non_italics,
        )
        self.text_align = "top-left"

    def save_im_text(self, text=""):
        im = og.sim.viewer_camera._get_obs()[0]['rgb'][:, :, :3]
        # if (isinstance(self.env, og.DialogPrimitivesEnv)
        #         or isinstance(self.env, og.DialogPrimitivesEnvV2)):
        #     obs, obs_info = self.env.get_obs(flatten_im=False, update_dialog_idx=False)
        #     im_robot_view = obs['image']
        #     if len(im_robot_view.shape) == 3 and im_robot_view.shape[0] == 3:
        #          # It is currently (3, H, W). Make into (H, W, 3)
        #          im_robot_view = im_robot_view.transpose(1, 2, 0)
        # else:
        obs, obs_info = self.env.get_obs()
        robot_name = self.env.robots[0].name
        im_robot_view = obs[robot_name][f'{robot_name}:left_eef_link:Camera:0']['rgb'][:, :, :3]
        # im_robot_view = obs['robot0']['robot0:eyes:Camera:0']['rgb'][:, :, :3]
        self.save_obs(im, im_robot_view, text)

    def save_obs(self, im_arr, im_arr_robot_view, text=""):
        def resize_im_arr(im_arr, downscale_factor=1):
            # assumes im_arr is (H, W, 3)
            im_arr = cv2.resize(
                im_arr,
                tuple([
                    int(x)
                    for x in (
                        np.array(im_arr.shape[:2][::-1])
                        // downscale_factor)]))
            return im_arr

        if th.is_tensor(im_arr):
            im_arr = im_arr.cpu().numpy()
        if th.is_tensor(im_arr_robot_view):
            im_arr_robot_view = im_arr_robot_view.cpu().numpy()

        im_arr = resize_im_arr(
            im_arr, self.vid_downscale_factor)
        im_arr_robot_view = resize_im_arr(im_arr_robot_view)
        im_arr_w_robot_view = self.overlay_robot_view_on_imgs(im_arr, im_arr_robot_view)
        im_arr_w_robot_view = self.maybe_add_text(im_arr_w_robot_view, text)
        self.ims.append(im_arr)
        self.robot_view_ims.append(im_arr_robot_view)
        self.ims_w_robot_view.append(im_arr_w_robot_view)

    def save_obs_batch(self, im_arr_list, im_arr_robot_view):
        assert len(im_arr_list) == len(im_arr_robot_view)
        for im_arr, im_arr_robot_view in zip(im_arr_list, im_arr_robot_view):
            self.save_obs(im_arr, im_arr_robot_view)

    def get_textbox_size(self, text, font):
        # Get image text size on dummy image
        im_dummy = Image.new(mode="P", size=(0, 0))
        draw_dummy = ImageDraw.Draw(im_dummy)
        _, _, text_w, text_h = draw_dummy.textbbox(
            (0, 0), text=text, font=self.fonts['italics'])
        return text_w, text_h

    def maybe_add_text(self, im_arr, new_text=""):
        if new_text:
            self.text_to_num_frames_remaining_map[new_text] = (
                self.num_frames_to_show_text)

        im = Image.fromarray(im_arr)
        im_w, im_h = im.size
        draw = ImageDraw.Draw(im)

        # Draw speedup
        speedup_text = f"{self.vid_speedup}x"
        text_w, text_h = self.get_textbox_size(speedup_text, self.fonts['non_italics'])
        draw.text(
            (0.98 * (im_w - text_w), 0.98 * (im_h - text_h)),
            speedup_text, font=self.fonts['non_italics'], fill="white")

        # Refresh counters at the end
        keys_to_remove = []
        num_active_texts = len([
            (text, num_frames_left)
            for text, num_frames_left in self.text_to_num_frames_remaining_map.items()
            if num_frames_left > 0])

        texts_to_draw = []
        active_text_idx = 0
        for text, num_frames_left in (
                self.text_to_num_frames_remaining_map.items()):

            # No longer an active word; was placed on enough frames already.
            if num_frames_left <= 0:
                keys_to_remove.append(text)
                continue

            # Center the text
            # Get image text size on dummy image
            text_w, text_h = self.get_textbox_size(text, self.fonts['italics'])
            if self.text_align == "center":
                x = 0.5 * (im_w - text_w)
                y = 0.5 * (im_h - (
                    num_active_texts - active_text_idx) * self.line_spacing * text_h)
            elif self.text_align == "top-left":
                x = 0.05 * im_w
                y = 0.05 * im_h + active_text_idx * self.line_spacing * text_h
            else:
                raise NotImplementedError

            color = (
                (0xf4, 0xe5, 0xbb) if "robot" in text.lower()
                else (0xbb, 0xca, 0xf4))
            # draw.text((x, y), text, font=self.fonts['italics'], fill=color)
            texts_to_draw.append((text, (x, y), (text_w, text_h), self.fonts['italics'], color))

            active_text_idx += 1
            self.text_to_num_frames_remaining_map[text] -= 1

        if len(texts_to_draw) > 0:
            # Draw translucent box behind text
            draw = ImageDraw.Draw(im, "RGBA")
            pad = 0.2 * self.text_font_size

            top_text_xy_pos = texts_to_draw[0][1]
            bottom_text_xy_pos = texts_to_draw[-1][1]
            largest_width_text_pos_box_size = max(
                [(pos, box_size) for _, pos, box_size, _, _ in texts_to_draw],
                key=lambda pos_size: pos_size[1][0])
            largest_width_text_xy_pos, largest_width_text_xy_size = largest_width_text_pos_box_size

            bottom_text_xy_size = texts_to_draw[-1][2]
            min_x, min_y = np.array([largest_width_text_xy_pos[0], top_text_xy_pos[1]]) - pad
            max_x = largest_width_text_xy_pos[0] + largest_width_text_xy_size[0] + pad
            max_y = bottom_text_xy_pos[1] + bottom_text_xy_size[1] + pad
            draw.rectangle(((min_x, min_y), (max_x, max_y)), fill=(0, 0, 0, 64))

            # Actually draw the text over the box
            for text, pos, _, font, color in texts_to_draw:
                draw.text(pos, text, font=font, fill=color)

        for key in keys_to_remove:
            self.text_to_num_frames_remaining_map.pop(key)

        return np.asarray(im)

    def overlay_robot_view_on_imgs(self, im, robot_view_im):
        assert im.shape[:2] > robot_view_im.shape[:2]
        im_w_robot_view = np.copy(im)
        h, w = robot_view_im.shape[:2]
        # Make patch in upper right
        im_w_robot_view[:h, -w:, :] = robot_view_im
        return im_w_robot_view

    def clear_ims(self):
        self.ims = []
        self.robot_view_ims = []
        self.ims_w_robot_view = []

    def make_video(self, prefix):
        imgs = np.array(self.ims_w_robot_view)
        self.save_video(imgs, prefix)
        self.clear_ims()

    def save_video(self, imgs, prefix):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        frame_height, frame_width = imgs[0].shape[:2]
        now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_dir = os.path.join(self.out_dir, prefix)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        out_path = os.path.join(out_dir, f"{prefix}_{now}.mp4")
        out = cv2.VideoWriter(
            out_path,
            fourcc, 30.0, (frame_width, frame_height))
        for i, frame in enumerate(imgs):
            if i % self.vid_speedup != 0:
                # Drop the frames to cause speedup.
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame)
        out.release()
        cv2.destroyAllWindows()
        print(f"saved video to {out_path}")