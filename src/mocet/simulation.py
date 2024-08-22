import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def get_rotation(vector, rx, ry, rz):
    # rx, ry, rz
    rx_matrix = np.array([[1, 0, 0],
                          [0, np.cos(rx), -np.sin(rx)],
                          [0, np.sin(rx), np.cos(rx)]])

    ry_matrix = np.array([[np.cos(-ry), 0, np.sin(-ry)],
                           [0, 1, 0],
                           [-np.sin(-ry), 0, np.cos(-ry)]])

    rz_matrix = np.array([[np.cos(-rz), -np.sin(-rz), 0],
                         [np.sin(-rz), np.cos(-rz), 0],
                         [0, 0, 1]])

    rotation_matrix = rx_matrix @ ry_matrix @ rz_matrix
    return rotation_matrix @ vector


def get_translation(vector, tx, ty, tz):
    translation_matrix = np.array([[1, 0, 0, tx],  # left is left
                                   [0, 1, 0, -ty], # front is back
                                   [0, 0, 1, -tz], # downside is upside
                                   [0, 0, 0, 1]])
    vector = translation_matrix @ np.append(vector, [1], axis=0)
    return vector[:3]


def get_transform(vector, tx, ty, tz, rx, ry, rz):
    vector = get_translation(vector, tx, ty, tz)
    vector = get_rotation(vector, rx, ry, rz)
    return vector


def normalize(vector):
    return vector / np.linalg.norm(vector)


class Scene():
    def __init__(self,
                 width,
                 height,
                 objects,
                 light=False):
        self.w = np.array(width)
        self.h = np.array(height)
        self.ratio = width / height
        self.objects = objects
        self.image = np.zeros((height, width, 3))
        self.ka = 0.1
        self.alpha = 30
        self.light = light

    def add_light(self,
                  light_position,
                  light_color=(1, 1, 1)):
        self.Lo = np.array(light_position)
        self.Lc = np.array(light_color)
        self.light = True

    def get_normal(self, intersection, object):
        return normalize(intersection - object.position)

    def add_camera(self,
                   camera_position,
                   camera_direction,
                   camera_tilt=None,
                   depth_limit=3):
        if camera_tilt is None:
            camera_tilt = [0, 0, 1]
        self.Co = np.array(camera_position)
        self.Cd = normalize(np.array(camera_direction) - self.Co)
        self.Cu = normalize(np.array(camera_tilt))
        self.Cr = normalize(np.cross(self.Cd, self.Cu))
        self.hFOV = 15
        self.depth = depth_limit
        self.pixel_w = 2 * np.tan(np.radians(self.hFOV / 2)) / self.w
        self.pixel_h = self.pixel_w

    def intersection(self, ray_origin, ray_direction, object):
        if object.type == 'sphere':
            # Ray-Sphere intersection
            O = ray_origin
            D = ray_direction
            P0 = object.position
            R = object.radius

            a = np.dot(D, D)  # always 1
            b = 2 * np.dot(D, O - P0)
            c = np.dot(O - P0, O - P0) - R * R
            discriminant = b * b - 4 * a * c
            if discriminant > 0:  # two roots
                t1 = (-b + np.sqrt(discriminant)) / (2.0 * a)
                t2 = (-b - np.sqrt(discriminant)) / (2.0 * a)

                if t1 > 0 and t2 > 0:  # find closest intersection
                    distance = np.min([t1, t2])
                elif t1 <= 0 and t2 <= 0:  # no intersection
                    distance = np.inf
                else:
                    distance = np.max([t1, t2])

            elif discriminant == 0:  # one root
                t = -b / (2 * a)
                distance = t

            elif discriminant < 0:  # no root
                distance = np.inf

            intersection = O + distance * ray_direction
            return distance, intersection

    def trace(self, ray_origin, ray_direction):
        # Step 1: Find the closest object
        min_distance = np.inf
        closest_object = None
        closest_object_idx = None
        closest_intersection = None  # closest intersection point
        for o, obj in enumerate(self.objects):
            distance, intersection = self.intersection(ray_origin, ray_direction, obj)
            if distance < min_distance:
                min_distance = distance
                closest_object = obj
                closest_object_idx = o
                closest_intersection = intersection
        if min_distance == np.inf:  # no object
            return np.array([0.4, 0.4, 0.4])

        # Step 2: Get properties of the closest object
        if self.light:
            ambient_color = closest_object.color
            diffuse_color = closest_object.color
            specular_color = self.Lc
            N = self.get_normal(closest_intersection, closest_object)
            L = normalize(self.Lo - closest_intersection)
            V = normalize(ray_origin - closest_intersection)
            H = normalize(L + V)

            # Step 3: Find if the intersection point is shadowed or not.
            distance_to_other_objects = []
            for o, obj in enumerate(self.objects):
                if o != closest_object_idx:
                    distance, _ = self.intersection(closest_intersection + N * .0001,
                                                    L,
                                                    obj)
                    distance_to_other_objects.append(distance)

            # Step 4: Apply Blinn-Phong reflection model
            color = self.ka * ambient_color  # add ambient
            if np.min(distance_to_other_objects) < np.inf:  # intersection point is shadowed
                return color
            color += closest_object.kd * max(np.dot(N, L), 0) * diffuse_color  # add diffuse
            color += closest_object.ks * max(np.dot(N, H), 0) ** self.alpha * specular_color  # add specular
            return color
        else:
            # Step 2: Get properties of the closest object
            color = closest_object.color
            # Step 3:
            return color

    def render(self):
        for x in range(self.w):
            for y in range(self.h):
                dx = self.pixel_w * (x - self.w / 2)
                dy = - self.pixel_h * (y - self.h / 2)

                O = self.Co  # Origin of ray
                D = normalize(self.Cd + dx * self.Cr + dy * self.Cu)  # Direction of ray

                color = self.trace(O, D)
                self.image[y, x] = np.clip(color, 0, 1)

    def draw(self, dpi=100):
        plt.figure(dpi=dpi, figsize=(3.2, 2.4))
        plt.imshow(self.image)
        plt.xticks([])
        plt.yticks([])
        plt.show()

    def save(self, filename):
        image = np.flip(self.image, axis=1)
        image = 255.0 * image
        image = image.astype(np.uint8)
        im = Image.fromarray(image).resize((320, 240), Image.NEAREST)
        im.save(filename)


class Sphere():
    def __init__(self, position, radius, color, diffuse_k, specular_k):
        self.type = 'sphere'
        self.position = position
        self.radius = radius
        self.color = np.array(color)
        self.kd = diffuse_k
        self.ks = specular_k


class Head():
    def __init__(self, headball_color=None):
        self.position = np.array([0, 0, 0])
        self.kd = 1.0
        self.ks = 1.0
        self.headball_radius = 0.0947
        if headball_color is None:
            self.headball_color = (0, 1, 0)
        else:
            self.headball_color = headball_color

        self.headball = Sphere(position=self.position,
                               radius=self.headball_radius,
                               color=self.headball_color,
                               diffuse_k=self.kd,
                               specular_k=self.ks)
        self.objects = [self.headball]

        self.eyeball_radius = 0.01185
        self.eyeball_color = (1, 1, 1)
        self.pupil_distance = 0.01
        self.pupil_radius = 0.0025
        self.pupil_color = (0, 0, 0)
        self.set_gaze()

        self.have_eyeball_left = False
        self.have_eyeball_right = False

    def add_eyeball_left(self, eyeball_position=np.array([-0.0315, 0.0847, 0.0037])):
        self.position_left = eyeball_position
        self.have_eyeball_left = True
        eyeball = Sphere(position=eyeball_position,
                         radius=self.eyeball_radius,
                         color=self.eyeball_color,
                         diffuse_k=self.kd,
                         specular_k=self.ks)

        # calculate gaze direction of left eyeball
        gaze_direction = normalize(self.gaze_target - eyeball_position)

        pupil_position = eyeball_position + gaze_direction * self.pupil_distance
        pupil = Sphere(position=pupil_position, radius=self.pupil_radius,
                       color=self.pupil_color, diffuse_k=self.kd, specular_k=self.ks)
        self.objects.append(eyeball)
        self.objects.append(pupil)

    def add_eyeball_right(self, eyeball_position=np.array([+0.0315, 0.0847, 0.0037])):
        self.position_right = eyeball_position
        self.have_eyeball_right = True
        eyeball = Sphere(position=eyeball_position,
                         radius=self.eyeball_radius,
                         color=self.eyeball_color,
                         diffuse_k=self.kd,
                         specular_k=self.ks)

        # calculate gaze direction of left eyeball
        gaze_direction = normalize(self.gaze_target - eyeball_position)
        pupil_position = eyeball_position + gaze_direction * self.pupil_distance
        pupil = Sphere(position=pupil_position, radius=self.pupil_radius,
                       color=self.pupil_color, diffuse_k=self.kd, specular_k=self.ks)
        self.objects.append(eyeball)
        self.objects.append(pupil)

    def motion(self, tx, ty, tz, rx, ry, rz):
        self.position = get_transform(self.position, tx, ty, tz, rx, ry, rz)
        self.headball = Sphere(position=self.position,
                               radius=self.headball_radius,
                               color=self.headball_color,
                               diffuse_k=self.kd,
                               specular_k=self.ks)
        self.objects = [self.headball]
        if self.have_eyeball_left:
            eyeball_position = get_transform(self.position_left, tx, ty, tz, rx, ry, rz)
            self.position_left = eyeball_position
            eyeball = Sphere(position=eyeball_position,
                             radius=self.eyeball_radius,
                             color=self.eyeball_color,
                             diffuse_k=self.kd,
                             specular_k=self.ks)

            # calculate gaze direction of left eyeball
            gaze_direction = normalize(self.gaze_target - eyeball_position)
            pupil_position = eyeball_position + gaze_direction * self.pupil_distance
            pupil = Sphere(position=pupil_position, radius=self.pupil_radius,
                           color=self.pupil_color, diffuse_k=self.kd, specular_k=self.ks)
            self.objects.append(eyeball)
            self.objects.append(pupil)

        if self.have_eyeball_right:
            eyeball_position = get_transform(self.position_right, tx, ty, tz, rx, ry, rz)
            self.position_right = eyeball_position
            eyeball = Sphere(position=eyeball_position,
                             radius=self.eyeball_radius,
                             color=self.eyeball_color,
                             diffuse_k=self.kd,
                             specular_k=self.ks)

            # calculate gaze direction of left eyeball
            gaze_direction = normalize(self.gaze_target - eyeball_position)
            pupil_position = eyeball_position + gaze_direction * self.pupil_distance
            pupil = Sphere(position=pupil_position, radius=self.pupil_radius,
                           color=self.pupil_color, diffuse_k=self.kd, specular_k=self.ks)
            self.objects.append(eyeball)
            self.objects.append(pupil)

    def set_gaze(self, gaze=np.array([0.0, 1.0, 0.0])):
        self.gaze_target = gaze

    def reset(self):
        self.objects = [self.headball]


class Screen():
    def __init__(self):
        # screen width in pixel
        self.screen_width_px = 1600
        self.screen_height_px = 1000

        # screen width in meter
        self.screen_width_m = 0.36
        self.screen_height_m = 0.23

        self.screen_distance = 1.0 + 0.0847 + 0.01

        self.screen_width_space = np.linspace(-self.screen_width_m / 2, +self.screen_width_m / 2,
                                              self.screen_width_px + 1)
        self.screen_height_space = np.linspace(+self.screen_height_m / 2, -self.screen_height_m / 2,
                                               self.screen_height_px + 1)

    def get_target(self, pixels):
        px_x = pixels[0]
        px_y = pixels[1]
        return np.array([self.screen_width_space[px_x], self.screen_distance, self.screen_height_space[px_y]])

def generate(motion_parameter,
               basis_parameter, save_directory,
               index = 0,
               calibration_onsets=None,
               calibration_points=None,
             render=True):

    if calibration_points is None:
        calibration_points = [24, 12]
    if calibration_onsets is None:
        calibration_onsets = [1, 494]
    duration = len(motion_parameter)
    calibration_coordinates = np.array([[200, 166], [200, 500], [200, 833],
                                        [600, 166], [600, 500], [600, 833],
                                        [1000, 166], [1000, 500], [1000, 833],
                                        [1400, 166], [1400, 500], [1400, 833]])
    calibration_order = [4, 11, 6, 2, 7, 0, 10, 5, 9, 8, 1, 3]
    center_fixation = np.array([800, 500])
    camera_resolution = [96, 72]
    camera_position = (-0.035, basis_parameter[2], 0.0037)
    camera_direction = (basis_parameter[0], -1, basis_parameter[1])
    camera_tilt = [np.sin(np.deg2rad(basis_parameter[3])), 0, np.cos(np.deg2rad(basis_parameter[3]))]

    head = Head()
    head.add_eyeball_left()
    original_position = head.position_left
    screen = Screen()

    calibration_i = 0
    validation_i = 0
    for tr in range(duration):
        if tr in np.arange(calibration_onsets[0], calibration_onsets[0]+calibration_points[0]):
            calibration_i = calibration_i % 12
            target = calibration_coordinates[calibration_order[calibration_i]]
            gaze_target = screen.get_target(np.array([target[0], target[1]]))
            calibration_i += 1
        elif tr in np.arange(calibration_onsets[1], calibration_onsets[1]+calibration_points[1]):
            validation_i = validation_i % 12
            target = calibration_coordinates[calibration_order[validation_i]]
            gaze_target = screen.get_target(np.array([target[0], target[1]]))
            validation_i += 1
        else:
            gaze_target = screen.get_target(np.array([center_fixation[0], center_fixation[1]]))

        head = Head(headball_color=(1,1,1))
        head.set_gaze(gaze=gaze_target)
        head.add_eyeball_left()

        tx = motion_parameter[tr,0] * 0.001 # millimeter to meter
        ty = motion_parameter[tr,1] * 0.001
        tz = motion_parameter[tr,2] * 0.001
        rx = motion_parameter[tr,3]
        ry = motion_parameter[tr,4]
        rz = motion_parameter[tr,5]
        head.motion(tx, ty, tz, rx, ry, rz)

        if render:
            scene = Scene(width=camera_resolution[0], height=camera_resolution[1], objects=head.objects)
            scene.add_camera(camera_position=camera_position, camera_direction=camera_direction, camera_tilt=camera_tilt)
            scene.render()
            scene.save(filename=f'{save_directory}/fig_{index + tr:09d}.png')
        head.reset()

    final_position = head.position_left
    direction = final_position - original_position
    length = np.linalg.norm(direction)

    angle = np.arccos(np.dot(normalize(np.array(camera_direction)), normalize(direction)))
    inplane_displacement = length * np.cos(angle - np.pi / 2)
    return inplane_displacement
    #return np.sqrt(np.sum([(original_position[i]-final_position[i])**2 for i in [0,2]]))