# Tên file: generate_perspective_data.py
# Mục đích: Tạo dataset unwarp ảnh chỉ với biến dạng phối cảnh (mô phỏng các góc chụp khác nhau).
# Giữ nguyên cấu trúc input/output và cơ chế tạo nhãn của script gốc.

import argparse
import sys, os
import pickle
import random
import numpy as np
import cv2
import time
from multiprocessing import Pool
import utils # Phải có file utils.py đi kèm

def get_datasets(dir_path):
    """Lấy danh sách các file trong một thư mục."""
    return os.listdir(dir_path)

class PerspectiveWarper(utils.BasePerturbed):
    """
    Lớp này chỉ thực hiện biến dạng phối cảnh để mô phỏng các góc chụp khác nhau
    của một tài liệu phẳng.
    """
    def __init__(self, path, bg_path, save_path, save_suffix):
        self.path = path
        self.bg_path = bg_path
        self.save_path = save_path
        self.save_suffix = save_suffix

    def generate_image(self, m, n, fiducial_points=16, relativeShift_position='relativeShift_v2'):
        """
        Hàm chính để tạo một cặp (ảnh biến dạng, nhãn), đã được tối ưu hóa.
        """
        try:
            # === BƯỚC 1: KHỞI TẠO VÀ CHUẨN BỊ ẢNH GỐC ===
            origin_img = cv2.imread(self.path, flags=cv2.IMREAD_COLOR)
            if origin_img is None:
                print(f"Lỗi: Không thể đọc ảnh {self.path}")
                return

            save_img_shape = (1024, 960) # Sử dụng tuple (height, width) cho nhất quán
            enlarge_canvas_shape = (save_img_shape[0] * 3, save_img_shape[1] * 3)

            scale_factor = random.uniform(0.4, 0.8)
            
            im_h, im_w = origin_img.shape[:2]
            
            base_shrink_w = (save_img_shape[1] - random.randint(16, 128)) * scale_factor
            base_shrink_h = (save_img_shape[0] - random.randint(16, 128)) * scale_factor

            if im_w > im_h:
                new_w = int(base_shrink_w)
                new_h = int(im_h / im_w * new_w)
            else:
                new_h = int(base_shrink_h)
                new_w = int(im_w / im_h * new_h)

            if new_h < fiducial_points or new_w < fiducial_points:
                print(f"Cảnh báo: Kích thước ảnh sau khi scale quá nhỏ ({new_w}x{new_h}), bỏ qua ảnh {self.path}")
                return

            edge_padding = 3
            new_h -= (new_h - 2 * edge_padding) % (fiducial_points - 1)
            new_w -= (new_w - 2 * edge_padding) % (fiducial_points - 1)
            
            self.origin_img = cv2.resize(origin_img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            
            im_h, im_w = self.origin_img.shape[:2]
            im_h_coords = np.linspace(edge_padding, im_h - edge_padding, fiducial_points, dtype=np.int64)
            im_w_coords = np.linspace(edge_padding, im_w - edge_padding, fiducial_points, dtype=np.int64)
            im_y_grid, im_x_grid = np.meshgrid(im_w_coords, im_h_coords)

            # === BƯỚC 2: ĐẶT ẢNH LÊN CANVAS LỚN VÀ TẠO NHÃN BAN ĐẦU ===
            self.new_shape = enlarge_canvas_shape
            x_min, y_min, x_max, y_max = self.adjust_position_v2(0, 0, im_h, im_w, self.new_shape)
            
            # Tạo các map nguồn
            synthesis_img_map = np.zeros((self.new_shape[0], self.new_shape[1], 3), dtype=np.uint8)
            synthesis_img_map[x_min:x_max, y_min:y_max] = self.origin_img
            
            origin_pixel_position = np.mgrid[0:im_h, 0:im_w].transpose(1, 2, 0)
            synthesis_label_map = np.zeros((self.new_shape[0], self.new_shape[1], 2), dtype=np.float32)
            synthesis_label_map[x_min:x_max, y_min:y_max] = origin_pixel_position

            foreORbackground_label_map = np.zeros(self.new_shape, dtype=np.uint8)
            foreORbackground_label_map[x_min:x_max, y_min:y_max] = 255 # Dùng 255 để nội suy tốt hơn

            # === BƯỚC 3: TẠO MA TRẬN BIẾN ĐỔI PHỐI CẢNH ===
            pts1 = np.float32([[y_min, x_min], [y_max-1, x_min], [y_min, x_max-1], [y_max-1, x_max-1]])

            perspective_strength_h = (x_max - x_min) * random.uniform(0.05, 0.35)
            perspective_strength_w = (y_max - y_min) * random.uniform(0.05, 0.35)

            # Vòng lặp tìm góc phù hợp để tránh biến dạng quá mức
            while True:
                pts2 = pts1 + (np.random.rand(4, 2) - 0.5) * np.array([perspective_strength_w, perspective_strength_h])
                pts2 = pts2.astype(np.float32)
                
                a0, a1, a2, a3 = self.get_angle_4(pts2)
                min_angle, max_angle = 70, 110
                if (min_angle < a0 < max_angle) and (min_angle < a1 < max_angle) and \
                   (min_angle < a2 < max_angle) and (min_angle < a3 < max_angle):
                    break
            
            M = cv2.getPerspectiveTransform(pts1, pts2)

            # === BƯỚC 4: ÁP DỤNG BIẾN ĐỔI BẰNG cv2.warpPerspective (SỬA LỖI VÀ TĂNG TỐC) ===
            dsize = (self.new_shape[1], self.new_shape[0]) # (width, height) cho OpenCV

            # Biến đổi ảnh màu
            synthesis_perturbed_img = cv2.warpPerspective(synthesis_img_map, M, dsize,
                                                          flags=cv2.INTER_CUBIC,
                                                          borderMode=cv2.BORDER_CONSTANT,
                                                          borderValue=(0, 0, 0))
            # Biến đổi bản đồ tọa độ
            synthesis_perturbed_label = cv2.warpPerspective(synthesis_label_map, M, dsize,
                                                            flags=cv2.INTER_LINEAR,
                                                            borderMode=cv2.BORDER_CONSTANT,
                                                            borderValue=(0, 0, 0))
            # Biến đổi mặt nạ foreground/background
            foreORbackground_label = cv2.warpPerspective(foreORbackground_label_map, M, dsize,
                                                         flags=cv2.INTER_NEAREST)
            
            # Chuyển mặt nạ về dạng 0 và 1
            foreORbackground_label = (foreORbackground_label > 128).astype(np.float32)

            # === BƯỚC 5: HẬU XỬ LÝ VÀ LƯU KẾT QUẢ ===
            rows, cols = np.where(foreORbackground_label == 1)
            if len(rows) == 0:
                print(f"Lỗi: Không có tiền cảnh nào được tạo cho {self.path}")
                return
            
            crop_x_min, crop_x_max = np.min(rows), np.max(rows)
            crop_y_min, crop_y_max = np.min(cols), np.max(cols)
            
            center_x, center_y = (crop_x_min + crop_x_max) // 2, (crop_y_min + crop_y_max) // 2
            final_h, final_w = save_img_shape[0], save_img_shape[1]
            
            start_row = max(0, center_x - final_h // 2)
            end_row = start_row + final_h
            start_col = max(0, center_y - final_w // 2)
            end_col = start_col + final_w
            
            self.synthesis_perturbed_img = synthesis_perturbed_img[start_row:end_row, start_col:end_col]
            self.synthesis_perturbed_label = synthesis_perturbed_label[start_row:end_row, start_col:end_col]
            self.foreORbackground_label = foreORbackground_label[start_row:end_row, start_col:end_col]
            self.new_shape = self.synthesis_perturbed_img.shape[:2]

            # Tính toán tọa độ điểm mốc (fiducial points) trên ảnh đã biến đổi
            src_fiducial_points = np.stack([im_y_grid, im_x_grid], axis=-1).reshape(-1, 1, 2).astype(np.float32)
            src_fiducial_points[:, 0, 0] += y_min # add y offset
            src_fiducial_points[:, 0, 1] += x_min # add x offset
            dst_fiducial_points = cv2.perspectiveTransform(src_fiducial_points, M)
            fiducial_points_coordinate = dst_fiducial_points.reshape(fiducial_points, fiducial_points, 2)
            
            fiducial_points_coordinate[:, :, 0] -= start_col # adjust for crop
            fiducial_points_coordinate[:, :, 1] -= start_row # adjust for crop
            
            pixel_position_final = np.mgrid[0:self.new_shape[0], 0:self.new_shape[1]].transpose(1, 2, 0)
            if relativeShift_position == 'relativeShift_v2':
                self.synthesis_perturbed_label -= pixel_position_final
            
            # Áp dụng mặt nạ
            self.synthesis_perturbed_label *= self.foreORbackground_label[..., np.newaxis]
            
            # Ghép nền
            perturbed_bg_img = cv2.imread(self.bg_path, flags=cv2.IMREAD_COLOR)
            if perturbed_bg_img is None:
                print(f"Lỗi: Không thể đọc ảnh nền {self.bg_path}")
                return
            
            perturbed_bg_img = cv2.resize(perturbed_bg_img, (self.new_shape[1], self.new_shape[0]))
            
            # Sử dụng np.where để ghép nền, nhanh và rõ ràng hơn
            background_mask = (self.foreORbackground_label == 0)[..., np.newaxis]
            final_image = np.where(background_mask, perturbed_bg_img, self.synthesis_perturbed_img)

            # Áp dụng các hiệu ứng cuối cùng
            final_image = self.HSV_v1(final_image.astype(np.uint8))
            if self.is_perform(0.2, 0.8):
                final_image = cv2.GaussianBlur(final_image, (3, 3), 0)

            final_image = np.clip(final_image, 0, 255).astype(np.uint8)
            
            # === LƯU KẾT QUẢ ===
            perfix_ = f'{self.save_suffix}_{m}_{n}'
            
            synthesis_perturbed_data = {
                'image': final_image, # Lưu ảnh cuối cùng
                'label': self.synthesis_perturbed_label, # Đổi tên cho nhất quán
                'fiducial_points': fiducial_points_coordinate,
                'segment': np.array(((im_h - 2*edge_padding) // (fiducial_points - 1), 
                                     (im_w - 2*edge_padding) // (fiducial_points - 1)))
            }
            
            cv2.imwrite(os.path.join(self.save_path, 'png', f'{perfix_}.png'), final_image)
            
            # Vẽ điểm mốc để debug
            img_with_fiducials = final_image.copy()
            for point in fiducial_points_coordinate.reshape(-1, 2):
                cv2.circle(img_with_fiducials, (int(point[0]), int(point[1])), 2, (0, 0, 255), -1)
            cv2.imwrite(os.path.join(self.save_path, 'fiducial_points', f'{perfix_}.png'), img_with_fiducials)
            
            # Lưu file pickle
            with open(os.path.join(self.save_path, 'color', f'{perfix_}.gw'), 'wb') as f:
                # Sử dụng protocol cao hơn để lưu file hiệu quả hơn
                pickle.dump(synthesis_perturbed_data, f, protocol=pickle.HIGHEST_PROTOCOL)

            print(f"Đã tạo thành công: {perfix_}.gw")

        except Exception as e:
            print(f"Xảy ra lỗi khi xử lý {self.path} với chỉ số {m}_{n}: {e}")
            import traceback
            traceback.print_exc()


def create_folders(save_path):
    """Tạo các thư mục đầu ra nếu chưa tồn tại."""
    os.makedirs(os.path.join(save_path, 'color'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'fiducial_points'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'png'), exist_ok=True)

def worker_process(args_tuple):
    """Hàm bao bọc để Pool.map có thể gọi."""
    img_path, bg_path, save_path, save_suffix, m, n, fiducial_points = args_tuple
    warper = PerspectiveWarper(img_path, bg_path, save_path, save_suffix)
    warper.generate_image(m, n, fiducial_points=fiducial_points)

def main(args):
    """Hàm chính điều phối quá trình tạo dữ liệu."""
    path = args.path
    bg_path = args.bg_path
    save_path = args.output_path
    
    create_folders(save_path)

    save_suffix = os.path.basename(os.path.dirname(path))
    all_img_paths = get_datasets(path)
    all_bg_paths = get_datasets(bg_path)

    if not all_img_paths:
        print(f"Không tìm thấy ảnh nào trong: {path}")
        return
    if not all_bg_paths:
        print(f"Không tìm thấy ảnh nền nào trong: {bg_path}")
        return

    begin_train = time.time()
    fiducial_points = 31 # Số điểm mốc

    tasks = []
    for m, img_name in enumerate(all_img_paths):
        for n in range(args.sys_num):
            img_path_full = os.path.join(path, img_name)
            bg_path_chosen = os.path.join(bg_path, random.choice(all_bg_paths))
            
            task_args = (img_path_full, bg_path_chosen, save_path, save_suffix, m, n, fiducial_points)
            tasks.append(task_args)

    # Sử dụng đa xử lý để tăng tốc
    with Pool(processes=args.num_workers) as process_pool:
        process_pool.map(worker_process, tasks)

    total_time = time.time() - begin_train
    print(f"\nHoàn thành tạo {len(tasks)} ảnh trong {total_time:.2f} giây.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tạo dữ liệu unwarp với biến dạng phối cảnh')
    parser.add_argument('--path', default='./scan/', type=str, help='Đường dẫn đến thư mục chứa ảnh gốc.')
    parser.add_argument('--bg_path', default='./background/', type=str, help='Đường dẫn đến thư mục chứa ảnh nền.')
    parser.add_argument('--output_path', default='./output/', type=str, help='Đường dẫn để lưu kết quả.')
    parser.add_argument('--sys_num', default=5, type=int, help='Số lượng phiên bản biến dạng cho mỗi ảnh gốc.')
    parser.add_argument('--num_workers', default=4, type=int, help='Số lượng tiến trình song song.')
    
    args = parser.parse_args()
    main(args)