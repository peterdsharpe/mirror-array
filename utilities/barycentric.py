import aerosandbox.numpy as np


def bary_to_cart(
        weights_bary: np.ndarray,
        vertices_cart: np.ndarray,
):
    weights_bary = np.array(weights_bary)
    vertices_cart = np.array(vertices_cart)
    points_cart = np.sum(
        weights_bary.reshape((-1, 1)) * vertices_cart,
        axis=0
    )
    return points_cart


def cart_to_bary(
        points_cart: np.ndarray,
        vertices_cart: np.ndarray
):
    points_cart = np.array(points_cart)
    vertices_cart = np.array(vertices_cart)

    if len(points_cart.shape) > 1:
        return np.array([
            cart_to_bary(
                points_cart=point_cart,
                vertices_cart=vertices_cart,
            )
            for point_cart in points_cart
        ])
    weights_bary = np.linalg.solve(
        A=np.array([
            vertices_cart[:, 0],
            vertices_cart[:, 1],
            [1, 1, 1]
        ]),
        b=[
            points_cart[0],
            points_cart[1],
            1
        ]
    )

    return weights_bary


if __name__ == '__main__':
    # print(
    #     bary_to_cart(
    #         weights_bary=[1 / 3, 1 / 3, 1 / 3],
    #         vertices_cart=[
    #             [0, 0],
    #             [1, 0],
    #             [0, 1]
    #         ]
    #     )
    # )
    verts = [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
    ]
    print(cart_to_bary(
        points_cart=[
            [1, 1, 0],
            [0, 0, 0]
        ],
        vertices_cart=verts
    ))
