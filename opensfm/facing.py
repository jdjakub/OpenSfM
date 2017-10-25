import numpy as np


def find_rotation_matrix_between_bases(basis_one, basis_two):
    """
    Finds a matrix to rotate between bases, where each orthogonal basis is only defined by the first two vectors
    The resulting change of basis matrix will convert vectors in basis_one to basis_two.
    Explicitly: If R is our returned matrix, R * basis_one = basis_two

    >>> basis_one = [np.array([1/np.sqrt(2),1/np.sqrt(2),0.0]), np.array([1/np.sqrt(2),-1/np.sqrt(2),0.0])]
    >>> basis_two = [np.array([1/np.sqrt(3)] * 3), np.array([0.0, 1/np.sqrt(2), -1/np.sqrt(2)])]
    >>> transform = find_rotation_matrix_between_bases(basis_one, basis_two)
    >>> np.allclose(transform.dot(basis_one[0]), basis_two[0])
    True
    >>> np.allclose(transform.dot(basis_one[1]), basis_two[1])
    True
    """

    transform_one = find_rotation_matrix_from_standard_basis(*basis_one)
    transform_two = find_rotation_matrix_from_standard_basis(*basis_two)
    transform_matrix = transform_two.dot(np.linalg.inv(transform_one))
    return transform_matrix


def find_rotation_matrix_from_standard_basis(vector_one, vector_two):
    """ Returns rotation matrix that rotates given vectors to standard basis vectors

    The net effect of this is that if the vectors given are the first two elements of an orthogonal basis, the returned
    matrix can be used as a change of basis matrix that moves us from the standard basis to our new basis
    """
    assert np.isclose(0, vector_one.dot(vector_two))

    normalized_one = vector_one / np.linalg.norm(vector_one)
    normalized_two = vector_two / np.linalg.norm(vector_two)

    vector_three = np.cross(normalized_one, normalized_two)
    transform_matrix = np.column_stack((normalized_one, normalized_two, vector_three))
    return transform_matrix


def make_first_camera_face_second(pose1, pose2):
    # Takes in opensfm reconstruction

    look_vector = pose2.get_origin() - pose1.get_origin()
    look_vector /= np.linalg.norm(look_vector)

    up_vector = np.array([0.0, 0.0, 1.0])
    camera_x = np.cross(look_vector, up_vector)
    camera_y = np.cross(look_vector, camera_x)

    basis_one = [look_vector, camera_y]
    basis_two = [np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0])]

    rotation_matrix = find_rotation_matrix_between_bases(basis_one, basis_two)

    origin = pose1.get_origin()
    pose1.set_rotation_matrix(rotation_matrix)
    pose1.set_origin(origin)
