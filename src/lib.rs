use std::collections::HashSet;
use std::error::Error;
use std::f64::consts::PI;
use std::fs;
use std::mem;

#[derive(Clone, Debug)]
/// Type of the data.
pub enum InstanceType {
    /// Data for a symmetric traveling salesperson problem.
    Tsp,
    /// Data for an asymmetric traveling salesperson problem.
    Atsp,
    /// Data for a sequential ordering problem.
    Sop,
    /// Hamiltonian cycle problem data.
    Hcp,
    /// Capacitated vehicle routing problem data.
    Cvrp,
    /// A collection of torus.
    Tours,
    /// Other problem data.
    Other(String),
}

#[derive(Clone, Copy, Debug)]
/// How the edge weights (or distances) are given.
pub enum EdgeWeightType {
    /// Weights are listed explicitly.
    Explicit,
    /// Weights are Euclidean distances in 2-D.
    Euc2d,
    /// Weights are Euclidean distances in 3-D.
    Euc3d,
    /// Weights are maximum distances in 2-D.
    Max2d,
    /// Weights are maximum distances in 3-D.
    Max3d,
    /// Weights are manhattan distances in 2-D.
    Man2d,
    /// Weights are manhattan distances in 3-D.
    Man3d,
    /// Weights are Euclidean distances in 2-D rounded up.
    Ceil2d,
    /// Weights are geographical distances.
    Geo,
    /// Special distance function for problems att48 and att532.
    Att,
    /// Special distance function for crystallography problems (Version 1).
    Xray1,
    /// Special distance function for crystallography problems (Version 2).
    Xray2,
    /// There is a special distance function documented elsewhere.
    Special,
}

#[derive(Clone, Copy, Debug)]
/// Describes the format of the edge weights.
pub enum EdgeWeightFormat {
    /// Weights are given by a function.
    Function,
    /// Weights are given by a full matrix.
    FullMatrix,
    /// Upper triangular matrix (row-wise without diagonal entries).
    UpperRow,
    /// Lower triangular matrix (row-wise without diagonal entries).
    LowerRow,
    /// Upper triangular matrix (row-wise including diagonal entries).
    UpperDiagRow,
    /// Lower triangular matrix (row-wise including diagonal entries).
    LowerDiagRow,
    /// Upper triangular matrix (column-wise without diagonal entries).
    UpperCol,
    /// Lower triangular matrix (column-wise without diagonal entries).
    LowerCol,
    /// Upper triangular matrix (column-wise including diagonal entries).
    UpperDiagCol,
    /// Lower triangular matrix (column-wise including diagonal entries).
    LowerDiagCol,
}

#[derive(Clone, Copy)]
enum EdgeDataFormat {
    EdgeList,
    AdjList,
}

#[derive(Clone, Copy)]
enum NodeCoordType {
    TwodCoords,
    ThreedCoords,
}

#[derive(PartialEq)]
enum DisplayDataType {
    Coord,
    Twod,
    No,
}

#[derive(Clone, Debug)]
/// Node coordinates.
pub enum NodeCoords {
    /// Nodes are specified by coordinates in 2-D.
    /// The first element of each tuple is the node number, the second and third elements are the x and y coordinates.
    Twod(Vec<(usize, f64, f64)>),
    /// Nodes are specified by coordinates in 3-D.
    /// The first element of each tuple is the node number, the second, third and fourth elements are the x, y, and z coordinates.
    Threed(Vec<(usize, f64, f64, f64)>),
}

#[derive(Clone, Debug)]
/// Edges of a graph.
pub enum EdgeData {
    /// Edges are given by a list of node pairs.
    EdgeList(Vec<(usize, usize)>),
    /// Edges are given by adjacency lists.
    /// The first element of each tuple is the node number, the second element is a list of adjacent nodes.
    AdjList(Vec<(usize, Vec<usize>)>),
}

#[derive(Clone, Debug)]
/// How a graphical display of the nodes can be obtained.
pub enum DisplayData {
    /// Display is generated from the node coordinates.
    CoordDisplay,
    /// Explicit coordinates in 2-D.
    /// The first element of each tuple is the node number, the second and third elements are the x and y coordinates.
    TwodDisplay(Vec<(usize, f64, f64)>),
}

#[derive(Clone, Debug)]
/// An instance of a TSP or related problem.
pub struct Instance {
    /// Name of the instance.
    pub name: String,
    /// Type of the data.
    pub instance_type: InstanceType,
    /// Additional comments.
    pub comment: String,
    /// For a TSP or ATSP, the dimension is the number of its nodes.
    /// For a CVRP, it is the total number of nodes and depots.
    /// For a TOUR file, it is the dimension of the corresponding problem.
    pub dimension: usize,
    /// The truck capacity.
    pub capacity: Option<i32>,
    /// How the edge weights (or distances) are given.
    pub edge_weight_type: EdgeWeightType,
    /// Describes the format of the edge weights if they are given explicitly.
    pub edge_weight_format: Option<EdgeWeightFormat>,
    /// Maximum route duration.
    pub distance: Option<f64>,
    /// Service time.
    pub service_time: Option<f64>,
    /// The number of commodities at each node.
    pub demand_dimension: Option<usize>,
    /// Node coordinates.
    /// (which, for example may be used for either graphical display or for computing the edge weights).
    pub node_coords: Option<NodeCoords>,
    /// A list of possible alternate depot nodes.
    pub depots: Option<Vec<usize>>,
    /// The demands of all nodes.
    /// The first element of each tuple is the node number, the second element is a list of possibly multi-dimensional demands.
    pub demands: Option<Vec<(usize, Vec<i32>)>>,
    /// Edges of a graph, if the graph is not complete.
    pub edge_data: Option<EdgeData>,
    /// Edges required to appear in each solution to the problem.
    pub fixed_edges: Option<Vec<(usize, usize)>>,
    /// How a graphical display of the nodes can be obtained.
    pub display_data: Option<DisplayData>,
    /// A collection of tours.
    pub tours: Option<Vec<Vec<usize>>>,
    /// Edge weights in a matrix format specified by `edge_weight_format`.
    pub edge_weights: Option<Vec<Vec<i32>>>,
}

fn parse_instance_type(value: &str) -> InstanceType {
    match value {
        "TSP" => InstanceType::Tsp,
        "ATSP" => InstanceType::Atsp,
        "SOP" => InstanceType::Sop,
        "HCP" => InstanceType::Hcp,
        "CVRP" => InstanceType::Cvrp,
        "TOUR" => InstanceType::Tours,
        other => InstanceType::Other(other.to_string()),
    }
}

fn parse_edge_weight_type(value: &str) -> Result<EdgeWeightType, Box<dyn Error>> {
    match value {
        "EXPLICIT" => Ok(EdgeWeightType::Explicit),
        "EUC_2D" => Ok(EdgeWeightType::Euc2d),
        "EUC_3D" => Ok(EdgeWeightType::Euc3d),
        "MAX_2D" => Ok(EdgeWeightType::Max2d),
        "MAX_3D" => Ok(EdgeWeightType::Max3d),
        "MAN_2D" => Ok(EdgeWeightType::Man2d),
        "MAN_3D" => Ok(EdgeWeightType::Man3d),
        "CEIL_2D" => Ok(EdgeWeightType::Ceil2d),
        "GEO" => Ok(EdgeWeightType::Geo),
        "ATT" => Ok(EdgeWeightType::Att),
        "XRAY1" => Ok(EdgeWeightType::Xray1),
        "XRAY2" => Ok(EdgeWeightType::Xray2),
        "SPECIAL" => Ok(EdgeWeightType::Special),
        _ => Err(format!("Unsupported edge weight type: {}", value).into()),
    }
}

fn parse_edge_weight_format(value: &str) -> Result<EdgeWeightFormat, Box<dyn Error>> {
    match value {
        "FUNCTION" => Ok(EdgeWeightFormat::Function),
        "FULL_MATRIX" => Ok(EdgeWeightFormat::FullMatrix),
        "UPPER_ROW" => Ok(EdgeWeightFormat::UpperRow),
        "LOWER_ROW" => Ok(EdgeWeightFormat::LowerRow),
        "UPPER_DIAG_ROW" => Ok(EdgeWeightFormat::UpperDiagRow),
        "LOWER_DIAG_ROW" => Ok(EdgeWeightFormat::LowerDiagRow),
        "UPPER_COL" => Ok(EdgeWeightFormat::UpperCol),
        "LOWER_COL" => Ok(EdgeWeightFormat::LowerCol),
        "UPPER_DIAG_COL" => Ok(EdgeWeightFormat::UpperDiagCol),
        "LOWER_DIAG_COL" => Ok(EdgeWeightFormat::LowerDiagCol),
        _ => Err(format!("Unsupported edge weight format: {}", value).into()),
    }
}

fn parse_edge_data_format(value: &str) -> Result<EdgeDataFormat, Box<dyn Error>> {
    match value {
        "EDGE_LIST" => Ok(EdgeDataFormat::EdgeList),
        "ADJ_LIST" => Ok(EdgeDataFormat::AdjList),
        _ => Err(format!("Unsupported edge data format: {}", value).into()),
    }
}

fn parse_node_coord_type(value: &str) -> Result<NodeCoordType, Box<dyn Error>> {
    match value {
        "TWOD_COORDS" => Ok(NodeCoordType::TwodCoords),
        "THREED_COORDS" => Ok(NodeCoordType::ThreedCoords),
        _ => Err(format!("Unsupported node coordinate type: {}", value).into()),
    }
}

fn parse_display_data_type(value: &str) -> Result<DisplayDataType, Box<dyn Error>> {
    match value {
        "COORD_DISPLAY" => Ok(DisplayDataType::Coord),
        "TWOD_DISPLAY" => Ok(DisplayDataType::Twod),
        "NO_DISPLAY" => Ok(DisplayDataType::No),
        _ => Err(format!("Unsupported display data type: {}", value).into()),
    }
}

type TwodCoords = Vec<(usize, f64, f64)>;

fn parse_2d_coords(
    lines: &mut impl Iterator<Item = String>,
    dimension: usize,
    line_number: &mut usize,
    section_name: &str,
) -> Result<TwodCoords, Box<dyn Error>> {
    let mut coords = Vec::with_capacity(dimension);
    let mut processed = HashSet::new();

    for _ in 0..dimension {
        let line = lines
            .next()
            .ok_or(format!("Unexpected EOF while parsing {}", section_name))?;
        *line_number += 1;

        let mut parts = line.split_whitespace();
        let node = parts
            .next()
            .ok_or(format!(
                "Missing node number in line {} ({})",
                line_number, section_name,
            ))?
            .parse()?;
        let x = parts
            .next()
            .ok_or(format!(
                "Missing x-coordinate in line {} of ({})",
                line_number, section_name
            ))?
            .parse()?;
        let y = parts
            .next()
            .ok_or(format!(
                "Missing y-coordinate in line {} of ({})",
                line_number, section_name
            ))?
            .parse()?;

        if processed.contains(&node) {
            return Err(format!(
                "Duplicate node number: {} in line {} (NODE_COORD_SECTION)",
                node, line_number
            )
            .into());
        }

        processed.insert(node);
        coords.push((node, x, y));
    }

    Ok(coords)
}

type ThreedCoords = Vec<(usize, f64, f64, f64)>;

fn parse_3d_coords(
    lines: &mut impl Iterator<Item = String>,
    dimension: usize,
    line_number: &mut usize,
) -> Result<ThreedCoords, Box<dyn Error>> {
    let mut coords = Vec::with_capacity(dimension);
    let mut processed = HashSet::new();

    for _ in 0..dimension {
        let line = lines
            .next()
            .ok_or("Unexpected EOF while parsing NODE_COORD_SECTION")?;
        *line_number += 1;

        let mut parts = line.split_whitespace();
        let node = parts
            .next()
            .ok_or(format!(
                "Missing node number in line {} of NODE_COORD_SECTION",
                line_number,
            ))?
            .parse()?;
        let x = parts
            .next()
            .ok_or(format!(
                "Missing x-coordinate in line {} of NODE_COORD_SECTION",
                line_number,
            ))?
            .parse()?;
        let y = parts
            .next()
            .ok_or(format!(
                "Missing y-coordinate in line {} of NODE_COORD_SECTION",
                line_number,
            ))?
            .parse()?;
        let z = parts
            .next()
            .ok_or(format!(
                "Missing z-coordinate in line {} of NODE_COORD_SECTION",
                line_number,
            ))?
            .parse()?;

        if processed.contains(&node) {
            return Err(format!(
                "Duplicate node number: {} in line {} (NODE_COORD_SECTION)",
                node, line_number
            )
            .into());
        }

        processed.insert(node);
        coords.push((node, x, y, z));
    }

    Ok(coords)
}

fn parse_node_coords(
    lines: &mut impl Iterator<Item = String>,
    node_coord_type: NodeCoordType,
    dimension: usize,
    line_number: &mut usize,
) -> Result<NodeCoords, Box<dyn Error>> {
    match node_coord_type {
        NodeCoordType::TwodCoords => Ok(NodeCoords::Twod(parse_2d_coords(
            lines,
            dimension,
            line_number,
            "NODE_COORD_SECTION",
        )?)),
        NodeCoordType::ThreedCoords => Ok(NodeCoords::Threed(parse_3d_coords(
            lines,
            dimension,
            line_number,
        )?)),
    }
}

fn parse_depots(
    lines: &mut impl Iterator<Item = String>,
    line_number: &mut usize,
) -> Result<Vec<usize>, Box<dyn Error>> {
    let mut depots = Vec::new();

    for line in lines {
        *line_number += 1;

        let mut iter = line.split_whitespace();

        while let Some(digit) = iter.next() {
            if digit == "-1" {
                if let Some(text) = iter.next() {
                    return Err(
                        format!("Unexpected text {} after -1 in DEPOT_SECTION", text).into(),
                    );
                }

                return Ok(depots);
            }

            let depot = digit.parse()?;
            depots.push(depot);
        }
    }

    Err("Unexpected EOF while parsing DEPOT_SECTION".into())
}

type Demands = Vec<(usize, Vec<i32>)>;

fn parse_demands(
    lines: &mut impl Iterator<Item = String>,
    dimension: usize,
    demand_dimension: usize,
    line_number: &mut usize,
) -> Result<Demands, Box<dyn Error>> {
    let mut demands = Vec::with_capacity(dimension);
    let mut processed = HashSet::new();

    for _ in 0..dimension {
        let line = lines
            .next()
            .ok_or("Unexpected EOF while parsing DEMAND_SECTION")?;
        *line_number += 1;

        let mut iter = line.split_whitespace();
        let node = iter
            .next()
            .ok_or(format!(
                "Missing node number in line {} (DEMAND_SECTION)",
                line_number
            ))?
            .parse::<usize>()?;

        let mut node_demands = Vec::with_capacity(demand_dimension);

        for j in 0..demand_dimension {
            let demand = iter
                .next()
                .ok_or(format!(
                    "Missing {}-th demand in line {} (DEMAND_SECTION)",
                    j + 1,
                    line_number
                ))?
                .parse::<i32>()?;

            node_demands.push(demand);
        }

        if processed.contains(&node) {
            return Err(format!(
                "Duplicate node number: {} in line {} (DEMAND_SECTION)",
                node, line_number
            )
            .into());
        }

        processed.insert(node);
        demands.push((node, node_demands));
    }

    Ok(demands)
}

fn parse_edge_list(
    lines: &mut impl Iterator<Item = String>,
    line_number: &mut usize,
) -> Result<Vec<(usize, usize)>, Box<dyn Error>> {
    let mut edge_list = Vec::new();

    for line in lines {
        *line_number += 1;

        if line.trim() == "-1" {
            return Ok(edge_list);
        }

        let mut parts = line.split_whitespace();
        let from = parts
            .next()
            .ok_or(format!(
                "Missing from node in line {} (EDGE_DATA_SECTION)",
                line_number
            ))?
            .parse()?;
        let to = parts
            .next()
            .ok_or(format!(
                "Missing to node in line {} (EDGE_DATA_SECTION)",
                line_number
            ))?
            .parse()?;

        edge_list.push((from, to));
    }

    Err("Unexpected EOF while parsing EDGE_DATA_SECTION".into())
}

type AdjList = Vec<(usize, Vec<usize>)>;

fn parse_adj_list(
    lines: &mut impl Iterator<Item = String>,
    line_number: &mut usize,
) -> Result<AdjList, Box<dyn Error>> {
    let mut adj_list = Vec::new();
    let mut from_node = None;
    let mut to_nodes = Vec::new();
    let mut processed = HashSet::new();

    for line in lines {
        *line_number += 1;
        let mut iter = line.split_whitespace();

        while let Some(digit) = iter.next() {
            if let Some(node) = from_node {
                if digit == "-1" {
                    if processed.contains(&node) {
                        return Err(format!(
                            "Duplicate node number: {} in line {} (EDGE_DATA_SECTION)",
                            node, line_number
                        )
                        .into());
                    }

                    processed.insert(node);
                    let mut tmp_nodes = Vec::new();
                    mem::swap(&mut tmp_nodes, &mut to_nodes);
                    adj_list.push((node, tmp_nodes));
                    from_node = None;
                } else {
                    to_nodes.push(digit.parse()?);
                }
            } else if digit == "-1" {
                if let Some(text) = iter.next() {
                    return Err(
                        format!("Unexpected text {} after -1 in EDGE_DATA_SECTION", text).into(),
                    );
                }

                return Ok(adj_list);
            } else {
                from_node = Some(digit.parse()?);
            }
        }
    }

    Err("Unexpected EOF while parsing EDGE_DATA_SECTION".into())
}

fn parse_edge_data(
    lines: &mut impl Iterator<Item = String>,
    edge_data_type: EdgeDataFormat,
    line_number: &mut usize,
) -> Result<EdgeData, Box<dyn Error>> {
    match edge_data_type {
        EdgeDataFormat::EdgeList => Ok(EdgeData::EdgeList(parse_edge_list(lines, line_number)?)),
        EdgeDataFormat::AdjList => Ok(EdgeData::AdjList(parse_adj_list(lines, line_number)?)),
    }
}

fn parse_tours(
    lines: &mut impl Iterator<Item = String>,
    line_number: &mut usize,
) -> Result<Vec<Vec<usize>>, Box<dyn Error>> {
    let mut tours = Vec::new();
    let mut current_tour = Vec::new();

    for line in lines {
        *line_number += 1;
        let mut iter = line.split_whitespace();

        while let Some(digit) = iter.next() {
            if digit == "-1" {
                if current_tour.is_empty() {
                    if let Some(text) = iter.next() {
                        return Err(
                            format!("Unexpected text {} after -1 in TOUR_SECTION", text).into()
                        );
                    }

                    return Ok(tours);
                } else {
                    let mut tmp_tour = Vec::new();
                    mem::swap(&mut tmp_tour, &mut current_tour);
                    tours.push(tmp_tour);
                }
            }

            let node = digit.parse()?;
            current_tour.push(node);
        }
    }

    Err("Unexpected EOF while parsing TOUR_SECTION".into())
}

fn adjust_full_matrix(matrix: &mut [Vec<i32>], last_element: i32) -> i32 {
    let mut last_element = last_element;

    for row in matrix.iter_mut().rev() {
        let first_element = row.remove(0);
        row.push(last_element);
        last_element = first_element;
    }

    last_element
}

fn parse_full_matrix(
    lines: &mut impl Iterator<Item = String>,
    dimension: usize,
    line_number: &mut usize,
) -> Result<Vec<Vec<i32>>, Box<dyn Error>> {
    let mut matrix = Vec::with_capacity(dimension);

    if dimension == 0 {
        return Ok(matrix);
    }

    let mut row = Vec::with_capacity(dimension);
    let mut i = 0;
    let mut j = 0;

    for line in lines {
        *line_number += 1;
        let mut iter = line.split_whitespace();

        while let Some(weight) = iter.next() {
            let weight = weight.parse::<i32>()?;
            row.push(weight);
            j += 1;

            if j == dimension {
                let mut tmp_row = Vec::with_capacity(dimension);
                mem::swap(&mut tmp_row, &mut row);
                matrix.push(tmp_row);
                i += 1;
                j = 0;

                if i == dimension {
                    if let Some(text) = iter.next() {
                        if let Some(text) = iter.next() {
                            return Err(format!(
                                "Unexpected text {} after full matrix in EDGE_WEIGHT_SECTION",
                                text,
                            )
                            .into());
                        }

                        let last_element = text.parse::<i32>()?;
                        let first_element = adjust_full_matrix(&mut matrix, last_element) as usize;

                        if first_element == dimension {
                            return Ok(matrix);
                        } else {
                            return Err(format!(
                                    "The first element {} is different from dimension {} for full matrix in EDGE_WEIGHT_SECTION",
                                    first_element, dimension
                                )
                                .into());
                        }
                    }

                    return Ok(matrix);
                }
            }
        }
    }

    Err(format!(
        "EDGE_WEIGHT_SECTION contains only {} < {} values",
        matrix.into_iter().flatten().count() + row.len(),
        dimension * dimension
    )
    .into())
}

fn parse_upper_row(
    lines: &mut impl Iterator<Item = String>,
    dimension: usize,
    line_number: &mut usize,
) -> Result<Vec<Vec<i32>>, Box<dyn Error>> {
    let mut matrix = Vec::with_capacity(dimension);

    if dimension == 0 {
        return Ok(matrix);
    }

    let mut n_columns = dimension - 1;

    let mut row = Vec::with_capacity(n_columns);

    if n_columns == 0 {
        matrix.push(row);

        return Ok(matrix);
    }

    let mut i = 1;
    let mut j = 0;

    for line in lines {
        *line_number += 1;
        let mut iter = line.split_whitespace();

        while let Some(weight) = iter.next() {
            let weight = weight.parse::<i32>()?;
            row.push(weight);
            j += 1;

            if j == n_columns {
                let mut tmp_row = Vec::with_capacity(n_columns);
                mem::swap(&mut tmp_row, &mut row);
                matrix.push(tmp_row);
                i += 1;
                j = 0;
                n_columns -= 1;

                if i == dimension {
                    if let Some(test) = iter.next() {
                        return Err(format!(
                            "Unexpected text {} after upper row in EDGE_WEIGHT_SECTION",
                            test
                        )
                        .into());
                    }

                    return Ok(matrix);
                }
            }
        }
    }

    Err(format!(
        "EDGE_WEIGHT_SECTION contains only {} < {} values",
        matrix.into_iter().flatten().count() + row.len(),
        dimension * (dimension - 1) / 2
    )
    .into())
}

fn parse_lower_row(
    lines: &mut impl Iterator<Item = String>,
    dimension: usize,
    line_number: &mut usize,
) -> Result<Vec<Vec<i32>>, Box<dyn Error>> {
    let mut matrix = Vec::with_capacity(dimension);

    if dimension == 0 {
        return Ok(matrix);
    }

    matrix.push(Vec::new());

    if dimension == 1 {
        return Ok(matrix);
    }

    let mut n_columns = 1;
    let mut row = Vec::with_capacity(n_columns);
    let mut i = 1;
    let mut j = 0;

    for line in lines {
        *line_number += 1;
        let mut iter = line.split_whitespace();

        while let Some(weight) = iter.next() {
            let weight = weight.parse::<i32>()?;
            row.push(weight);
            j += 1;

            if j == n_columns {
                let mut tmp_row = Vec::with_capacity(n_columns);
                mem::swap(&mut tmp_row, &mut row);
                matrix.push(tmp_row);
                i += 1;
                j = 0;
                n_columns += 1;

                if i == dimension {
                    if let Some(test) = iter.next() {
                        return Err(format!(
                            "Unexpected text {} after lower row in EDGE_WEIGHT_SECTION",
                            test
                        )
                        .into());
                    }

                    return Ok(matrix);
                }
            }
        }
    }

    Err(format!(
        "EDGE_WEIGHT_SECTION contains only {} < {} values",
        matrix.into_iter().flatten().count() + row.len(),
        dimension * (dimension - 1) / 2
    )
    .into())
}

fn parse_upper_diag_row(
    lines: &mut impl Iterator<Item = String>,
    dimension: usize,
    line_number: &mut usize,
) -> Result<Vec<Vec<i32>>, Box<dyn Error>> {
    let mut matrix = Vec::with_capacity(dimension);

    if dimension == 0 {
        return Ok(matrix);
    }

    let mut n_columns = dimension;
    let mut row = Vec::with_capacity(n_columns);
    let mut i = 0;
    let mut j = 0;

    for line in lines {
        *line_number += 1;
        let mut iter = line.split_whitespace();

        while let Some(weight) = iter.next() {
            let weight = weight.parse::<i32>()?;
            row.push(weight);
            j += 1;

            if j == n_columns {
                let mut tmp_row = Vec::with_capacity(n_columns);
                mem::swap(&mut tmp_row, &mut row);
                matrix.push(tmp_row);
                i += 1;
                j = 0;
                n_columns -= 1;

                if i == dimension {
                    if let Some(test) = iter.next() {
                        return Err(format!(
                            "Unexpected text {} after upper diag row in EDGE_WEIGHT_SECTION",
                            test
                        )
                        .into());
                    }

                    return Ok(matrix);
                }
            }
        }
    }

    Err(format!(
        "EDGE_WEIGHT_SECTION contains only {} < {} values",
        matrix.into_iter().flatten().count() + row.len(),
        dimension * (dimension + 1) / 2
    )
    .into())
}

fn parse_lower_diag_row(
    lines: &mut impl Iterator<Item = String>,
    dimension: usize,
    line_number: &mut usize,
) -> Result<Vec<Vec<i32>>, Box<dyn Error>> {
    let mut matrix = Vec::with_capacity(dimension);

    if dimension == 0 {
        return Ok(matrix);
    }

    let mut n_columns = 1;
    let mut row = Vec::with_capacity(n_columns);
    let mut i = 0;
    let mut j = 0;

    for line in lines {
        *line_number += 1;
        let mut iter = line.split_whitespace();

        while let Some(weight) = iter.next() {
            let weight = weight.parse::<i32>()?;
            row.push(weight);
            j += 1;

            if j == n_columns {
                let mut tmp_row = Vec::with_capacity(n_columns);
                mem::swap(&mut tmp_row, &mut row);
                matrix.push(tmp_row);
                i += 1;
                j = 0;
                n_columns += 1;

                if i == dimension {
                    if let Some(test) = iter.next() {
                        return Err(format!(
                            "Unexpected text {} after lower diag row in EDGE_WEIGHT_SECTION",
                            test
                        )
                        .into());
                    }

                    return Ok(matrix);
                }
            }
        }
    }

    Err(format!(
        "EDGE_WEIGHT_SECTION contains only {} < {} values",
        matrix.into_iter().flatten().count() + row.len(),
        dimension * (dimension + 1) / 2
    )
    .into())
}

fn parse_edge_weights(
    lines: &mut impl Iterator<Item = String>,
    edge_weight_format: EdgeWeightFormat,
    dimension: usize,
    line_number: &mut usize,
) -> Result<Vec<Vec<i32>>, Box<dyn Error>> {
    match edge_weight_format {
        EdgeWeightFormat::FullMatrix => parse_full_matrix(lines, dimension, line_number),
        EdgeWeightFormat::UpperRow | EdgeWeightFormat::LowerCol => {
            parse_upper_row(lines, dimension, line_number)
        }
        EdgeWeightFormat::LowerRow | EdgeWeightFormat::UpperCol => {
            parse_lower_row(lines, dimension, line_number)
        }
        EdgeWeightFormat::UpperDiagRow | EdgeWeightFormat::LowerDiagCol => {
            parse_upper_diag_row(lines, dimension, line_number)
        }
        EdgeWeightFormat::LowerDiagRow | EdgeWeightFormat::UpperDiagCol => {
            parse_lower_diag_row(lines, dimension, line_number)
        }
        other => Err(format!(
            "Edge weight format {:?} does not support EDGE_WEIGHT_SECTION",
            other
        )
        .into()),
    }
}

impl Instance {
    fn read_from_lines(
        lines: &mut impl Iterator<Item = String>,
    ) -> Result<Instance, Box<dyn Error>> {
        let mut name = String::from("");
        let mut instance_type = InstanceType::Other("".to_string());
        let mut comment = String::from("");
        let mut dimension = None;
        let mut capacity = None;
        let mut edge_weight_type = None;
        let mut edge_weight_format = None;
        let mut edge_data_format = None;
        let mut node_coord_type = None;
        let mut display_data_type = None;
        let mut distance = None;
        let mut service_time = None;
        let mut demand_dimension = None;

        let mut node_coords = None;
        let mut depots = None;
        let mut demands = None;
        let mut edge_data = None;
        let mut fixed_edges = None;
        let mut display_data = None;
        let mut tours = None;
        let mut edge_weights = None;

        let mut processed_keys = HashSet::new();

        let mut line_number = 0;

        while let Some(line) = lines.next() {
            line_number += 1;
            let trimmed = line.trim();

            if trimmed.is_empty() {
                continue;
            }

            if trimmed == "EOF" {
                break;
            }

            if let Some((key, value)) = trimmed.split_once(":") {
                let value = value.trim();
                let key = key.trim();

                if processed_keys.contains(key) {
                    return Err(format!("Duplicate key: {}", key).into());
                }

                match key {
                    "NAME" => {
                        name = value.to_string();
                        processed_keys.insert("NAME");
                        continue;
                    }
                    "TYPE" => {
                        instance_type = parse_instance_type(value);
                        processed_keys.insert("TYPE");
                        continue;
                    }
                    "COMMENT" => {
                        comment = value.to_string();
                        processed_keys.insert("COMMENT");
                        continue;
                    }
                    "DIMENSION" => {
                        dimension = Some(value.parse()?);
                        processed_keys.insert("DIMENSION");
                        continue;
                    }
                    "CAPACITY" => {
                        capacity = Some(value.parse()?);
                        processed_keys.insert("CAPACITY");
                        continue;
                    }
                    "EDGE_WEIGHT_TYPE" => {
                        edge_weight_type = Some(parse_edge_weight_type(value)?);
                        processed_keys.insert("EDGE_WEIGHT_TYPE");
                        continue;
                    }
                    "EDGE_WEIGHT_FORMAT" => {
                        edge_weight_format = Some(parse_edge_weight_format(value)?);
                        processed_keys.insert("EDGE_WEIGHT_FORMAT");
                        continue;
                    }
                    "EDGE_DATA_FORMAT" => {
                        edge_data_format = Some(parse_edge_data_format(value)?);
                        processed_keys.insert("EDGE_DATA_FORMAT");
                        continue;
                    }
                    "NODE_COORD_TYPE" => {
                        node_coord_type = Some(parse_node_coord_type(value)?);
                        processed_keys.insert("NODE_COORD_TYPE");
                        continue;
                    }
                    "DISPLAY_DATA_TYPE" => {
                        display_data_type = Some(parse_display_data_type(value)?);
                        processed_keys.insert("DISPLAY_DATA_TYPE");
                        continue;
                    }
                    "DISTANCE" => {
                        distance = Some(value.parse()?);
                        processed_keys.insert("DISTANCE");
                        continue;
                    }
                    "SERVICE_TIME" => {
                        service_time = Some(value.parse()?);
                        processed_keys.insert("SERVICE_TIME");
                        continue;
                    }
                    "DEMAND_DIMENSION" => {
                        demand_dimension = Some(value.parse()?);
                        processed_keys.insert("DEMAND_DIMENSION");
                        continue;
                    }
                    _ => {}
                }
            }

            let section_key_result = line.split(":").collect::<Vec<_>>();

            if section_key_result.len() == 1
                || section_key_result.len() == 2 && section_key_result[1].is_empty()
            {
                let key = section_key_result[0].trim();

                if processed_keys.contains(key) {
                    return Err(format!("Duplicate key: {}", key).into());
                }

                match key {
                    "NODE_COORD_SECTION" => match (node_coord_type, dimension) {
                        (_, None) => {
                            return Err(
                                "DIMENSION must be specified before NODE_COORD_SECTION".into()
                            );
                        }
                        (Some(node_coord_type), Some(dimension)) => {
                            node_coords = Some(parse_node_coords(
                                lines,
                                node_coord_type,
                                dimension,
                                &mut line_number,
                            )?);
                            processed_keys.insert("NODE_COORD_SECTION");
                            continue;
                        }
                        (None, Some(dimension)) => {
                            let result = parse_2d_coords(
                                lines,
                                dimension,
                                &mut line_number,
                                "NODE_COORD_SECTION",
                            );

                            if result.is_ok() {
                                node_coords = Some(NodeCoords::Twod(result?));
                                processed_keys.insert("NODE_COORD_SECTION");
                                continue;
                            }

                            let result = parse_3d_coords(lines, dimension, &mut line_number);

                            if result.is_ok() {
                                node_coords = Some(NodeCoords::Threed(result?));
                                processed_keys.insert("NODE_COORD_SECTION");
                                continue;
                            }

                            return Err("Failed to parse NODE_COORD_SECTION".into());
                        }
                    },
                    "DEPOT_SECTION" => {
                        depots = Some(parse_depots(lines, &mut line_number)?);
                        processed_keys.insert("DEPOT_SECTION");
                        continue;
                    }
                    "DEMAND_SECTION" => {
                        if let Some(dimension) = dimension {
                            if demand_dimension.is_none() {
                                demand_dimension = Some(1);
                            }

                            demands = Some(parse_demands(
                                lines,
                                dimension,
                                demand_dimension.unwrap(),
                                &mut line_number,
                            )?);
                            processed_keys.insert("DEMAND_SECTION");
                            continue;
                        } else {
                            return Err("DIMENSION must be specified before DEMAND_SECTION".into());
                        }
                    }
                    "EDGE_DATA_SECTION" => match edge_data_format {
                        Some(edge_data_format) => {
                            edge_data =
                                Some(parse_edge_data(lines, edge_data_format, &mut line_number)?);
                            processed_keys.insert("EDGE_DATA_SECTION");
                            continue;
                        }
                        None => {
                            return Err(
                                "EDGE_DATA_FORMAT must be specified before EDGE_DATA_SECTION"
                                    .into(),
                            );
                        }
                    },
                    "FIXED_EDGES_SECTION" => {
                        fixed_edges = Some(parse_edge_list(lines, &mut line_number)?);
                        processed_keys.insert("FIXED_EDGES_SECTION");
                        continue;
                    }
                    "DISPLAY_DATA_SECTION" => {
                        if display_data_type != Some(DisplayDataType::Twod) {
                            return Err("DISPLAY_DATA_TYPE must be TWOD_DISPLAY when DISPLAY_DATA_SECTION is used".into());
                        }

                        if let Some(dimension) = dimension {
                            display_data = Some(DisplayData::TwodDisplay(parse_2d_coords(
                                lines,
                                dimension,
                                &mut line_number,
                                "DISPLAY_DATA_SECTION",
                            )?));
                            processed_keys.insert("DISPLAY_DATA_SECTION");
                            continue;
                        } else {
                            return Err(
                                "DIMENSION must be specified before DISPLAY_DATA_SECTION".into()
                            );
                        }
                    }
                    "TOUR_SECTION" => {
                        tours = Some(parse_tours(lines, &mut line_number)?);
                        processed_keys.insert("TOUR_SECTION");
                        continue;
                    }
                    "EDGE_WEIGHT_SECTION" => {
                        if let Some(edge_weight_format) = edge_weight_format {
                            if let Some(dimension) = dimension {
                                edge_weights = Some(parse_edge_weights(
                                    lines,
                                    edge_weight_format,
                                    dimension,
                                    &mut line_number,
                                )?);
                                processed_keys.insert("EDGE_WEIGHT_SECTION");
                                continue;
                            } else {
                                return Err(
                                    "DIMENSION must be specified before EDGE_WEIGHT_SECTION".into(),
                                );
                            }
                        } else {
                            return Err(
                                "EDGE_WEIGHT_FORMAT must be specified before EDGE_WEIGHT_SECTION"
                                    .into(),
                            );
                        }
                    }
                    _ => {}
                }
            }

            return Err(format!("Failed to parse line {}: {}", line_number, line).into());
        }

        if dimension.is_none() {
            return Err("Missing dimension".into());
        }

        if edge_weight_type.is_none() {
            return Err("Missing edge weight type".into());
        }

        if display_data_type == Some(DisplayDataType::Coord) && node_coord_type.is_none() {
            return Err(
                "NODE_COORD_TYPE must be specified when DISPLAY_DATA_TYPE is COORD_DISPLAY".into(),
            );
        }

        if display_data_type.is_none() && node_coord_type.is_some() {
            display_data = Some(DisplayData::CoordDisplay);
        }

        Ok(Instance {
            name,
            instance_type,
            comment,
            dimension: dimension.unwrap(),
            capacity,
            edge_weight_type: edge_weight_type.unwrap(),
            edge_weight_format,
            distance,
            service_time,
            demand_dimension,
            node_coords,
            depots,
            demands,
            edge_data,
            fixed_edges,
            display_data,
            tours,
            edge_weights,
        })
    }

    /// Parses an instance from a string.
    pub fn parse(input: &str) -> Result<Instance, Box<dyn Error>> {
        let mut lines = input.lines().map(|line| line.to_string());

        Instance::read_from_lines(&mut lines)
    }

    /// Loads an instance from a file.
    pub fn load(filename: &str) -> Result<Instance, Box<dyn Error>> {
        let content = fs::read_to_string(filename)?;

        Instance::parse(&content)
    }
}

fn round(x: f64) -> i32 {
    (x + 0.5).trunc() as i32
}

fn euiclidean_distance_2d(x1: f64, y1: f64, x2: f64, y2: f64) -> i32 {
    let dx = x1 - x2;
    let dy = y1 - y2;

    round((dx * dx + dy * dy).sqrt())
}

fn euclidean_distance_3d(x1: f64, y1: f64, z1: f64, x2: f64, y2: f64, z2: f64) -> i32 {
    let dx = x1 - x2;
    let dy = y1 - y2;
    let dz = z1 - z2;

    round((dx * dx + dy * dy + dz * dz).sqrt())
}

fn manhattan_distance_2d(x1: f64, y1: f64, x2: f64, y2: f64) -> i32 {
    let dx = x1 - x2;
    let dy = y1 - y2;

    round(dx.abs() + dy.abs())
}

fn manhattan_distance_3d(x1: f64, y1: f64, z1: f64, x2: f64, y2: f64, z2: f64) -> i32 {
    let dx = x1 - x2;
    let dy = y1 - y2;
    let dz = z1 - z2;

    round(dx.abs() + dy.abs() + dz.abs())
}

fn maximum_distance_2d(x1: f64, y1: f64, x2: f64, y2: f64) -> i32 {
    let dx = x1 - x2;
    let dy = y1 - y2;

    round(dx.abs().max(dy.abs()))
}

fn maximum_distance_3d(x1: f64, y1: f64, z1: f64, x2: f64, y2: f64, z2: f64) -> i32 {
    let dx = x1 - x2;
    let dy = y1 - y2;
    let dz = z1 - z2;

    round(dx.abs().max(dy.abs()).max(dz.abs()))
}

fn radian(v: f64) -> f64 {
    let deg = round(v) as f64;
    let min = v - deg;

    PI * (deg + 5.0 * min / 3.0) / 180.0
}

fn geographical_distance(x1: f64, y1: f64, x2: f64, y2: f64) -> i32 {
    const RRR: f64 = 6378.388;

    let latitude1 = radian(x1);
    let longitude1 = radian(y1);
    let latitude2 = radian(x2);
    let longitude2 = radian(y2);

    let q1 = (longitude1 - longitude2).cos();
    let q2 = (latitude1 - latitude2).cos();
    let q3 = (latitude1 + latitude2).cos();

    (RRR * (0.5 * (1.0 + q1) * q2 - (1.0 - q1) * q3).acos() + 1.0).trunc() as i32
}

fn pseudo_euclidean_distance(x1: f64, y1: f64, x2: f64, y2: f64) -> i32 {
    let xd = x1 - x2;
    let yd = y1 - y2;
    let rij = ((xd * xd + yd * yd) / 10.0).sqrt();
    let tij = round(rij);

    if (tij as f64) < rij {
        tij + 1
    } else {
        tij
    }
}

fn ceiling_euclidean_distance_2d(x1: f64, y1: f64, x2: f64, y2: f64) -> i32 {
    let dx = x1 - x2;
    let dy = y1 - y2;

    (dx * dx + dy * dy).sqrt().ceil() as i32
}

fn xray1(x1: f64, y1: f64, z1: f64, x2: f64, y2: f64, z3: f64) -> i32 {
    let dx = (x1 - x2).abs();
    let dx_minus = (dx - 360.0).abs();
    let dx = if dx < dx_minus { dx } else { dx_minus };

    let dy = (y1 - y2).abs();
    let dz = (z1 - z3).abs();

    let distance = if dx > dy && dx > dz {
        dx
    } else if dy > dz {
        dy
    } else {
        dz
    };

    round(100.0 * distance)
}

fn xray2(x1: f64, y1: f64, z1: f64, x2: f64, y2: f64, z3: f64) -> i32 {
    const SX: f64 = 1.25;
    const SY: f64 = 1.5;
    const SZ: f64 = 1.15;

    let dx = (x1 - x2).abs();
    let dx_minus = (dx - 360.0).abs();
    let dx = if dx < dx_minus { dx } else { dx_minus };
    let dx = dx / SX;

    let dy = (y1 - y2).abs() / SY;
    let dz = (z1 - z3).abs() / SZ;

    let distance = if dx > dy && dx > dz {
        dx
    } else if dy > dz {
        dy
    } else {
        dz
    };

    round(100.0 * distance)
}

fn from_2d_coords_to_full_matrix<F>(coords: &[(usize, f64, f64)], f: F) -> Vec<Vec<i32>>
where
    F: Fn(f64, f64, f64, f64) -> i32,
{
    let dimension = coords.len();
    let mut matrix = vec![vec![0; dimension]; dimension];
    let mut coords = coords.to_vec();
    coords.sort_by_key(|(node, _, _)| *node);

    for i in 0..dimension {
        for j in 0..dimension {
            matrix[i][j] = f(coords[i].1, coords[i].2, coords[j].1, coords[j].2);
        }
    }

    matrix
}

fn from_3d_coords_to_full_matrix<F>(coords: &[(usize, f64, f64, f64)], f: F) -> Vec<Vec<i32>>
where
    F: Fn(f64, f64, f64, f64, f64, f64) -> i32,
{
    let dimension = coords.len();
    let mut matrix = vec![vec![0; dimension]; dimension];
    let mut coords = coords.to_vec();
    coords.sort_by_key(|(node, _, _, _)| *node);

    for i in 0..dimension {
        for j in 0..dimension {
            matrix[i][j] = f(
                coords[i].1,
                coords[i].2,
                coords[i].3,
                coords[j].1,
                coords[j].2,
                coords[j].3,
            );
        }
    }

    matrix
}

fn from_upper_row_to_full_matrix(matrix: &[Vec<i32>]) -> Vec<Vec<i32>> {
    let dimension = matrix.len();
    let mut full_matrix = vec![vec![0; dimension]; dimension];

    for i in 0..dimension {
        let n_columns = dimension - i - 1;

        for k in 0..n_columns {
            let j = i + 1 + k;

            full_matrix[i][j] = matrix[i][k];
            full_matrix[j][i] = matrix[i][k];
        }
    }

    full_matrix
}

fn from_lower_row_to_full_matrix(matrix: &[Vec<i32>]) -> Vec<Vec<i32>> {
    let dimension = matrix.len();
    let mut full_matrix = vec![vec![0; dimension]; dimension];

    for i in 0..dimension {
        for j in 0..i {
            full_matrix[i][j] = matrix[i][j];
            full_matrix[j][i] = matrix[i][j];
        }
    }

    full_matrix
}

fn from_upper_diag_row_to_full_matrix(matrix: &[Vec<i32>]) -> Vec<Vec<i32>> {
    let dimension = matrix.len();
    let mut full_matrix = vec![vec![0; dimension]; dimension];

    for i in 0..dimension {
        let n_columns = dimension - i;

        for k in 0..n_columns {
            let j = i + k;

            full_matrix[i][j] = matrix[i][k];
            full_matrix[j][i] = matrix[i][k];
        }
    }

    full_matrix
}

fn from_lower_diag_row_to_full_matrix(matrix: &[Vec<i32>]) -> Vec<Vec<i32>> {
    let dimension = matrix.len();
    let mut full_matrix = vec![vec![0; dimension]; dimension];

    for i in 0..dimension {
        for j in 0..=i {
            full_matrix[i][j] = matrix[i][j];
            full_matrix[j][i] = matrix[i][j];
        }
    }

    full_matrix
}

fn from_explicit_to_full_matrix(
    weights: &[Vec<i32>],
    format: &EdgeWeightFormat,
) -> Result<Vec<Vec<i32>>, Box<dyn Error>> {
    match format {
        EdgeWeightFormat::FullMatrix => Ok(weights.to_vec()),
        EdgeWeightFormat::UpperRow => Ok(from_upper_row_to_full_matrix(weights)),
        EdgeWeightFormat::LowerRow => Ok(from_lower_row_to_full_matrix(weights)),
        EdgeWeightFormat::UpperDiagRow => Ok(from_upper_diag_row_to_full_matrix(weights)),
        EdgeWeightFormat::LowerDiagRow => Ok(from_lower_diag_row_to_full_matrix(weights)),
        _ => Err(format!("Unsupported edge weight format: {:?}", format).into()),
    }
}

impl Instance {
    /// Returns the full distance matrix of the instance.
    pub fn get_full_distance_matrix(&self) -> Result<Vec<Vec<i32>>, Box<dyn Error>> {
        match self.edge_weight_type {
            EdgeWeightType::Special => Err("SPECIAL edge weight type is not supported".into()),
            EdgeWeightType::Explicit => {
                if let Some(format) = self.edge_weight_format {
                    if let Some(weights) = &self.edge_weights {
                        from_explicit_to_full_matrix(weights, &format)
                    } else {
                        Err(
                        "EDGE_WEIGHT_SECTION must be specified when EDGE_WEIGHT_TYPE is EXPLICIT"
                            .into(),
                    )
                    }
                } else {
                    Err(
                        "EDGE_WEIGHT_FORMAT must be specified when EDGE_WEIGHT_TYPE is EXPLICIT"
                            .into(),
                    )
                }
            }
            EdgeWeightType::Euc2d => {
                if let Some(NodeCoords::Twod(coords)) = &self.node_coords {
                    Ok(from_2d_coords_to_full_matrix(
                        coords,
                        euiclidean_distance_2d,
                    ))
                } else {
                    Err(
                        "NODE_COORD_SECTION must be TWOD_COORDS when EDGE_WEIGHT_TYPE is EUC_2D"
                            .into(),
                    )
                }
            }
            EdgeWeightType::Euc3d => {
                if let Some(NodeCoords::Threed(coords)) = &self.node_coords {
                    Ok(from_3d_coords_to_full_matrix(coords, euclidean_distance_3d))
                } else {
                    Err(
                        "NODE_COORD_SECTION must be THREED_COORDS when EDGE_WEIGHT_TYPE is EUC_3D"
                            .into(),
                    )
                }
            }
            EdgeWeightType::Max2d => {
                if let Some(NodeCoords::Twod(coords)) = &self.node_coords {
                    Ok(from_2d_coords_to_full_matrix(coords, maximum_distance_2d))
                } else {
                    Err(
                        "NODE_COORD_SECTION must be TWOD_COORDS when EDGE_WEIGHT_TYPE is MAX_2D"
                            .into(),
                    )
                }
            }
            EdgeWeightType::Max3d => {
                if let Some(NodeCoords::Threed(coords)) = &self.node_coords {
                    Ok(from_3d_coords_to_full_matrix(coords, maximum_distance_3d))
                } else {
                    Err(
                        "NODE_COORD_SECTION must be THREED_COORDS when EDGE_WEIGHT_TYPE is MAX_3D"
                            .into(),
                    )
                }
            }
            EdgeWeightType::Man2d => {
                if let Some(NodeCoords::Twod(coords)) = &self.node_coords {
                    Ok(from_2d_coords_to_full_matrix(coords, manhattan_distance_2d))
                } else {
                    Err(
                        "NODE_COORD_SECTION must be TWOD_COORDS when EDGE_WEIGHT_TYPE is MAN_2D"
                            .into(),
                    )
                }
            }
            EdgeWeightType::Man3d => {
                if let Some(NodeCoords::Threed(coords)) = &self.node_coords {
                    Ok(from_3d_coords_to_full_matrix(coords, manhattan_distance_3d))
                } else {
                    Err(
                        "NODE_COORD_SECTION must be THREED_COORDS when EDGE_WEIGHT_TYPE is MAN_3D"
                            .into(),
                    )
                }
            }
            EdgeWeightType::Ceil2d => {
                if let Some(NodeCoords::Twod(coords)) = &self.node_coords {
                    Ok(from_2d_coords_to_full_matrix(
                        coords,
                        ceiling_euclidean_distance_2d,
                    ))
                } else {
                    Err(
                        "NODE_COORD_SECTION must be TWOD_COORDS when EDGE_WEIGHT_TYPE is CEIL_2D"
                            .into(),
                    )
                }
            }
            EdgeWeightType::Geo => {
                if let Some(NodeCoords::Twod(coords)) = &self.node_coords {
                    Ok(from_2d_coords_to_full_matrix(coords, geographical_distance))
                } else {
                    Err(
                        "NODE_COORD_SECTION must be TWOD_COORDS when EDGE_WEIGHT_TYPE is GEO"
                            .into(),
                    )
                }
            }
            EdgeWeightType::Att => {
                if let Some(NodeCoords::Twod(coords)) = &self.node_coords {
                    Ok(from_2d_coords_to_full_matrix(
                        coords,
                        pseudo_euclidean_distance,
                    ))
                } else {
                    Err(
                        "NODE_COORD_SECTION must be TWOD_COORDS when EDGE_WEIGHT_TYPE is ATT"
                            .into(),
                    )
                }
            }
            EdgeWeightType::Xray1 => {
                if let Some(NodeCoords::Threed(coords)) = &self.node_coords {
                    Ok(from_3d_coords_to_full_matrix(coords, xray1))
                } else {
                    Err(
                        "NODE_COORD_SECTION must be THREED_COORDS when EDGE_WEIGHT_TYPE is XRAY1"
                            .into(),
                    )
                }
            }
            EdgeWeightType::Xray2 => {
                if let Some(NodeCoords::Threed(coords)) = &self.node_coords {
                    Ok(from_3d_coords_to_full_matrix(coords, xray2))
                } else {
                    Err(
                        "NODE_COORD_SECTION must be THREED_COORDS when EDGE_WEIGHT_TYPE is XRAY2"
                            .into(),
                    )
                }
            }
        }
    }
}
