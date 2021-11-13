import numpy as np
from ..lux.game import Game
from ..lux.constants import Constants


def game_state_to_global_input(game_state: Game, player: int):
    n_features = 15
    second_player = (player + 1) % 2
    state = np.zeros((n_features,))
    state[0] = game_state.turn / 360  # turn [0...1]
    state[1] = game_state.turn % 40 > 30  # is night
    state[2] = (game_state.turn % 40 - 10) / 30  # steps to night [0...1]
    state[3] = game_state.players[player].research_points / 200  # research [0...1] maybe more than 1
    state[4] = game_state.players[player].city_tile_count  # city tiles count [0...360]
    state[5] = len(game_state.players[player].units)  # units count [0...360]
    state[6] = len(game_state.players[player].cities)  # cities count [0...~100]
    state[7] = game_state.players[player].researched_coal()  # researched coal
    state[8] = game_state.players[player].researched_uranium()  # researched uranium
    state[9] = game_state.players[second_player].research_points / 200  # research [0...1] maybe more than 1
    state[10] = game_state.players[second_player].city_tile_count  # city tiles count [0...360]
    state[11] = len(game_state.players[second_player].units)  # units count [0...360]
    state[12] = len(game_state.players[second_player].cities)  # cities count [0...~100]
    state[13] = game_state.players[second_player].researched_coal()  # researched coal
    state[14] = game_state.players[second_player].researched_uranium()  # researched uranium

    return state


def game_state_to_input(game_state: Game, player: int, action_unit_id: str):
    n_features = 22
    state = np.zeros((32, 32, n_features), dtype=np.float32)  # 32x32x22 = 22528
    x_offset = (32 - game_state.map_width) // 2
    y_offset = (32 - game_state.map_height) // 2
    owner_index = 1
    city_fuel_index = 9
    city_light_upkeep_index = 10
    action_index = 12

    for yi in range(game_state.map_height):
        for xi in range(game_state.map_width):
            x = x_offset + xi
            y = y_offset + yi
            cell = game_state.map.get_cell(xi, yi)
            has_resource = cell.has_resource()
            has_citytile = cell.citytile is not None
            state[y][x][0] = 1  # it's real cell, it's not an offset
            state[y][x][owner_index] = has_citytile and cell.citytile.team == player  # player is owner of cell
            state[y][x][2] = has_resource  # has resource
            state[y][x][3] = has_resource and cell.resource.type == Constants.RESOURCE_TYPES.WOOD  # has wood
            state[y][x][4] = has_resource and cell.resource.type == Constants.RESOURCE_TYPES.COAL  # has coal
            state[y][x][5] = has_resource and cell.resource.type == Constants.RESOURCE_TYPES.URANIUM  # has uranium
            state[y][x][6] = has_resource and cell.resource.amount / 800  # resource amount [0...1]
            state[y][x][7] = has_citytile  # has city tile
            state[y][x][8] = has_citytile and cell.citytile.cooldown  # citytile cooldown [0...10]
            state[y][x][city_fuel_index] = 0  # city fuel
            state[y][x][city_light_upkeep_index] = 0  # city light upkeep
            state[y][x][11] = cell.road / 6  # road range [0...1]
            state[y][x][action_index] = has_citytile and cell.citytile.cityid == action_unit_id  # action unit or citytile

    def fill_units(units, is_owner):
        for unit in units:
            x = x_offset + unit.pos.x
            y = y_offset + unit.pos.y
            state[y][x][owner_index] = state[y][x][owner_index] or is_owner  # player is owner of cell
            state[y][x][action_index] = unit.id == action_unit_id  # action unit or citytile
            state[y][x][13] = unit.is_worker()                # is worker
            state[y][x][14] = unit.is_cart()                  # is cart
            state[y][x][15] = unit.cooldown                   # unit cooldown [0...3]
            state[y][x][16] = unit.cargo.wood                 # wood stored [0...2000]
            state[y][x][17] = unit.cargo.coal                 # coal stored [0...2000]
            state[y][x][18] = unit.cargo.uranium              # uranium stored [0...2000]
            state[y][x][19] = unit.can_build(game_state.map)  # can unit build
            state[y][x][20] = unit.can_act()                  # can unit act
            state[y][x][21] = unit.get_cargo_space_left()     # cargo space left [0...2000]

    def fill_cities(cities):
        for city_id in cities:
            for tile in cities[city_id].citytiles:
                x = x_offset + tile.pos.x
                y = y_offset + tile.pos.y
                state[y][x][city_fuel_index] = cities[city_id].fuel  # city fuel
                state[y][x][city_light_upkeep_index] = cities[city_id].light_upkeep  # city light upkeep

    fill_units(game_state.players[0].units, player == 0)
    fill_units(game_state.players[1].units, player == 1)
    fill_cities(game_state.players[0].cities)
    fill_cities(game_state.players[1].cities)

    return state
