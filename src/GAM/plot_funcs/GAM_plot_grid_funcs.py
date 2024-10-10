import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from typing import List
from pdf2image import convert_from_path
import re
import os
import math
from statsmodels.sandbox.stats.multicomp import multipletests
from loguru import logger
from pathlib import Path

src_path = Path.cwd() / Path("src")

PPL_PATH = src_path / Path("GAM/perplexity/models_names_with_ppl.csv")


def p2stars(p):
    # Define the function to map p-values to symbols
    if p < 0:
        return "()"
    elif p <= 0.001:
        return "***"
    elif p <= 0.01:
        return "**"
    elif p <= 0.05:
        return "*"
    else:
        return "(.)"


def add_symbol_by_p_val(values_df: pd.DataFrame):
    # Apply the function to create the 'symbol' column
    values_df["symbol"] = values_df["p_val"].apply(p2stars)
    return values_df


def add_x_y_vals_context_plot(
    values_df: pd.DataFrame, smooths_plot: bool, fillers: bool
):
    logger.info("Context Plot| add x y vals")
    x_col = "context_type"
    values_df = values_df.replace("standard_contxt", "standard_context")
    if fillers == "TRUE":
        x0 = "standard_context"
        x1 = "dots_control_context"
        x2 = "text_control_context"
    else:
        x0 = "standard_context"
        x1 = "regime_context"
        x2 = "prompt_regime_context"

    logger.info(f"Context Plot| {x0} {x1} {x2}")
    y_col = "reread_condition"
    y0 = 0
    y1 = 1
    z_col = "has_preview_condition"
    z0 = "Gathering"
    z1 = "Hunting"

    if smooths_plot:
        x_values = {x0: 155, x1: 685, x2: 1220}
        y_base = 85
        y_inc = 445
        y_values = {
            (y0, z0): y_base,
            (y0, z1): y_base + y_inc,
            (y1, z0): y_base + 2 * y_inc,
            (y1, z1): y_base + 3 * y_inc,
        }

        # Apply the mapping to create the new column
        values_df["x_surp"] = values_df[x_col].map(x_values)
        values_df["x_prev_surp"] = values_df["x_surp"]
        values_df["y_surp"] = (
            values_df[[y_col, z_col]].apply(tuple, axis=1).map(y_values)
        )
        values_df["y_prev_surp"] = values_df["y_surp"] + 35
        values_df.loc[
            values_df["linear"] == "nonlinear", ["x_surp", "x_prev_surp"]
        ] += 50
        values_df["x_context"] = values_df["x_surp"] + 100
        values_df["y_context"] = values_df["y_surp"]

        values_df["symbol_size"] = 26
        values_df["font_size"] = 26

    else:
        x_values = {x0: 180, x1: 685, x2: 1185}
        y_base = 80
        y_inc = 455
        y_values = {
            (y0, z0): y_base,
            (y0, z1): y_base + y_inc,
            (y1, z0): y_base + 2 * y_inc,
            (y1, z1): y_base + 3 * y_inc,
        }
        # Apply the mapping to create the new column
        # coordinates: p val linear vs nonlinear
        values_df["x"] = values_df[x_col].map(x_values)
        values_df["y"] = values_df[[y_col, z_col]].apply(tuple, axis=1).map(y_values)
        # coordinates: context name
        values_df["x_context"] = values_df["x"] + 70
        values_df["y_context"] = values_df["y"]

        values_df["symbol_size"] = 28
        values_df["font_size"] = 28

    # values_df.to_csv("debug.csv")
    return values_df


def add_x_y_vals_context_reread_plot(values_df: pd.DataFrame):
    logger.info("Context Plot| add x y reread_plot")
    x_col = "context_type"
    x0 = "regime_context"
    y_col = "reread_condition"
    y0 = 1
    z_col = "has_preview_condition"
    z0 = "Gathering"
    z1 = "Hunting"

    x_values = {
        x0: 200,
    }
    y_base = 90
    y_inc = 660
    y_values = {(y0, z0): y_base, (y0, z1): y_base + y_inc}

    # Apply the mapping to create the new column
    values_df["x_surp"] = values_df[x_col].map(x_values)
    values_df["x_prev_surp"] = values_df["x_surp"]
    values_df["y_surp"] = values_df[[y_col, z_col]].apply(tuple, axis=1).map(y_values)
    values_df["y_prev_surp"] = values_df["y_surp"] + 35
    values_df.loc[values_df["linear"] == "nonlinear", ["x_surp", "x_prev_surp"]] += 80
    values_df["x_context"] = values_df["x_surp"] + 600
    values_df["y_context"] = values_df["y_surp"]
    values_df["symbol_size"] = 36
    values_df["font_size"] = 36

    return values_df


def add_x_y_vals(
    values_df: pd.DataFrame,
    smooths_plot: bool = False,
    context_plot: bool = False,
    context_reread_plot: bool = False,
    fillers: bool = False,
):
    if context_plot:
        values_df = add_x_y_vals_context_plot(values_df, smooths_plot, fillers)
        return values_df
    if context_reread_plot:
        values_df = add_x_y_vals_context_reread_plot(values_df)
        return values_df

    if "has_preview_condition" in values_df.columns:
        x_col = "has_preview_condition"
        x0 = "Gathering"
        x1 = "Hunting"
        y_col = "reread_condition"
        y0 = 0
        y1 = 1
    if "critical_span_condition" in values_df.columns:
        x_col = "critical_span_condition"
        x0 = 0
        x1 = 1
        y_col = "reread_condition"
        y0 = 0
        y1 = 1
    if "reread_consecutive_condition" in values_df.columns:
        x_col = "has_preview_condition"
        x0 = "Gathering"
        x1 = "Hunting"
        y_col = "reread_consecutive_condition"
        y0 = 11
        y1 = 12

    if smooths_plot:
        # Define the x and y values based on conditions
        x_values = {x0: 180, x1: 670}
        y_base = {y0: 90, y1: 550}

        y_increment = 35

        # Compute the x and y columns - surp sumbols
        values_df["x_surp"] = values_df[x_col].map(x_values)
        values_df["y_surp"] = values_df.apply(lambda row: y_base[row[y_col]], axis=1)
        # Compute the x and y columns - prev surp symbols
        values_df["x_prev_surp"] = values_df[x_col].map(x_values)
        values_df["y_prev_surp"] = values_df.apply(
            lambda row: y_base[row[y_col]] + y_increment, axis=1
        )

        values_df.loc[
            values_df["linear"] == "nonlinear", ["x_surp", "x_prev_surp"]
        ] += 80
        values_df["symbol_size"] = 30
        values_df["font_size"] = 30

    else:
        # Define the x and y values based on conditions
        x_values = {x0: 190, x1: 680}
        y_base = {y0: 90, y1: 590}

        y_increment = 35
        values_df["symbol_size"] = 30
        values_df["font_size"] = 30

        # Compute the x and y columns
        values_df["x"] = values_df[x_col].map(x_values)
        values_df["y"] = values_df.apply(lambda row: y_base[row[y_col]], axis=1)
    return values_df


def get_group_by_condition(df):
    if "critical_span_condition" in df.columns:
        return ["critical_span_condition", "reread_condition", "linear"]
    elif "reread_consecutive_condition" in df.columns:
        return ["has_preview_condition", "reread_consecutive_condition", "linear"]
    elif "context" in df.columns:
        return [
            "has_preview_condition",
            "reread_condition",
            "linear",
            "context",
            "context_type",
        ]
    else:
        return ["has_preview_condition", "reread_condition", "linear"]


def correct_p_vals(df: pd.DataFrame, p_val_column: str):
    group_by_cols = get_group_by_condition(df)
    corrected_p_vals = pd.Series(index=df.index, dtype=float)
    grouped = df.groupby(group_by_cols)
    for _, group in grouped:
        p_vals = group[p_val_column]
        # Split p-values into those >= 0 and <= 1, and others
        valid_p_vals = p_vals[(p_vals >= 0) & (p_vals <= 1)]
        invalid_p_vals = p_vals[(p_vals < 0) | (p_vals > 1)]

        if not valid_p_vals.empty:
            _, corrected_p, _, _ = multipletests(valid_p_vals, method="bonferroni")
            corrected_p_vals[valid_p_vals.index] = corrected_p

        # Set invalid p-values to pd.NA
        corrected_p_vals[invalid_p_vals.index] = pd.NA
    return corrected_p_vals


def correct_p_vals_surp_cols(df: pd.DataFrame):
    # Apply the correction for both columns
    df["corrected_surp_p_val"] = correct_p_vals(df, "surp_p_val")
    df["corrected_prev_surp_p_val"] = correct_p_vals(df, "prev_surp_p_val")
    return df


def _valid_mean(series: pd.Series):
    valid_values = series[(series >= 0) & (series <= 1)]
    return valid_values.mean()


def _agg_valid_means(
    df: pd.DataFrame, group_by_cols: List[str], p_val_col: str, prev_p_val_col: str
):
    return (
        df.groupby(group_by_cols)[[p_val_col, prev_p_val_col]]
        .apply(lambda x: x.apply(_valid_mean))
        .reset_index()
    )


def get_smooths_p_vals_symbols(
    df: pd.DataFrame, smooths_p_vals_path: str, correct_by_multiple_test: bool = False
):
    p_val_col = "surp_p_val"
    prev_p_val_col = "prev_surp_p_val"
    group_by_cols = get_group_by_condition(df)
    logger.debug(f"Smooths Plot| agg p vals, group by: {group_by_cols}")

    if correct_by_multiple_test:
        logger.debug("Smooths Plot| using corrected p vals")
        df = correct_p_vals_surp_cols(df)
        save_corrected_smooths_p_vals_df(df, smooths_p_vals_path)
        p_val_col = "corrected_surp_p_val"
        prev_p_val_col = "corrected_prev_surp_p_val"

    agg_df = _agg_valid_means(df, group_by_cols, p_val_col, prev_p_val_col)
    agg_df["surp_symbol"] = agg_df[p_val_col].apply(p2stars)
    agg_df["prev_surp_symbol"] = agg_df[prev_p_val_col].apply(p2stars)
    return agg_df


def save_corrected_smooths_p_vals_df(
    corrected_df: pd.DataFrame, smooths_p_vals_path: str
):
    prefix_path = smooths_p_vals_path[: -len("_smooths_p_vals_df.csv")]
    new_file_path = prefix_path + "_smooths_p_vals_df_corrected.csv"
    corrected_df.to_csv(new_file_path)


def save_new_img(pil_image, img_path: str):
    folder_path, file_name = os.path.split(img_path)
    new_folder = "with_p_vals"
    new_folder_path = os.path.join(folder_path, new_folder)
    os.makedirs(new_folder_path, exist_ok=True)
    new_file_path = os.path.join(new_folder_path, file_name)
    pil_image.save(new_file_path)


def save_agg_symbols_df(
    smooths_p_vals_symbols: pd.DataFrame, smooths_p_val_df_path: str
):
    prefix_path = smooths_p_val_df_path[: -len("_smooths_p_vals_df.csv")]
    new_file_path = prefix_path + "_smooths_p_vals_df_with_symbols.csv"
    smooths_p_vals_symbols.to_csv(new_file_path)


def _replace_text(
    text: str,
    reread,
    col_names_df=pd.read_excel(
        src_path / Path("GAM/plot_funcs/GAM_plot_cols_names.xlsx")
    ),
):
    col_names_df["fill_col"] = col_names_df["context column"].replace("filler", "fill")
    #
    if "instFirstReadS" in text:
        condition = (col_names_df["context column"] == text) & (
            col_names_df["reread_condition"] == reread
        )
    else:
        condition = col_names_df["context column"] == text

    vals = col_names_df[condition]["name"].values
    if vals is not None and len(vals) > 0:
        new_text = vals[0]
        # logger.info(f"Plot| {text=}, {new_text=}")
    else:
        logger.warning(f"Plot| {text} is not in names df, {vals=}, {reread=}")
        new_text = vals
    if len(new_text) > 15:
        logger.info(f"Plot| {new_text} is too long")
    return new_text


def draw_on_dll_img(dll_img_path, dll_p_vals):
    # Open an existing image
    pil_image_lst = convert_from_path(
        dll_img_path
    )  # This returns a list even for a 1 page pdf
    pil_image = pil_image_lst[0]
    # Initialize the drawing context with the image object
    draw = ImageDraw.Draw(pil_image)
    # Load a font
    font_path = (
        "DejaVuSans-Bold.ttf"  # Replace with the path to your downloaded font file
    )

    dll_p_vals.to_csv("debug.csv")
    # Add the annotation to the image
    for _, row in dll_p_vals.iterrows():
        symbol_position = (row["x"], row["y"])
        symbol_size = row["font_size"]
        symbol = row["symbol"]
        color = "black"
        draw.text(
            symbol_position,
            symbol,
            fill=color,
            font=ImageFont.truetype(font_path, symbol_size),
        )

        if "context" in dll_p_vals.columns:
            text_position = (row["x_context"], row["y_context"])
            text_font_size = row["font_size"]
            text = row["context"]
            reread = row["reread_condition"]
            text = _replace_text(text, reread)
            if text == "NoName":
                continue
            color = "black"
            draw.text(
                text_position,
                text,
                fill=color,
                font=ImageFont.truetype(font_path, text_font_size),
            )

    # Save the image with the annotation
    save_new_img(pil_image, dll_img_path)


def draw_on_smooths_img(smooth_img_path, smooths_p_vals, context_reread_plot=False):
    # Open an existing image
    pil_image_lst = convert_from_path(
        smooth_img_path
    )  # This returns a list even for a 1 page pdf
    pil_image = pil_image_lst[0]
    # Initialize the drawing context with the image object
    draw = ImageDraw.Draw(pil_image)
    color_dict = {"linear": "#005CAB", "nonlinear": "#AF0038"}
    # Load a font
    font_path = (
        "DejaVuSans-Bold.ttf"  # Replace with the path to your downloaded font file
    )
    # Add the annotation to the image
    for _, row in smooths_p_vals.iterrows():
        for surp in ["surp"]:  # surp, prev_surp
            symbol_position = (row[f"x_{surp}"], row[f"y_{surp}"])
            symbol_size = row["symbol_size"]
            color = color_dict[row["linear"]]
            symbol = row[f"{surp}_symbol"]
            draw.text(
                symbol_position,
                symbol,
                fill=color,
                font=ImageFont.truetype(font_path, symbol_size),
            )

        if "context" in smooths_p_vals.columns and row["linear"] == "linear":
            text_position = (row["x_context"], row["y_context"])
            text_font_size = row["font_size"]
            text = row["context"]
            reread = row["reread_condition"]
            text = _replace_text(text, reread)
            if text == "NoName":
                continue
            color = "black"
            draw.text(
                text_position,
                text,
                fill=color,
                font=ImageFont.truetype(font_path, text_font_size),
            )

    # Save the image with the annotation
    save_new_img(pil_image, smooth_img_path)


def add_p_vals_symbols(
    dll_img_path,
    dll_p_vals_path,
    smooth_img_path,
    smooths_p_vals_path,
    context_plot=False,
    fillers=False,
):
    # Read the CSV file
    dll_p_vals = pd.read_csv(dll_p_vals_path)
    smooths_p_vals = pd.read_csv(smooths_p_vals_path)
    # Add x, y, symbol columns
    logger.info("DLL Plot | ----------------------")
    dll_p_vals = add_x_y_vals(
        dll_p_vals, smooths_plot=False, context_plot=context_plot, fillers=fillers
    )
    dll_p_vals = add_symbol_by_p_val(dll_p_vals)
    draw_on_dll_img(dll_img_path, dll_p_vals)
    # Add x,y columns, symbol columns
    logger.info("Smooths Plot | ----------------------")
    smooths_p_vals = get_smooths_p_vals_symbols(smooths_p_vals, smooths_p_vals_path)
    smooths_p_vals = add_x_y_vals(
        smooths_p_vals, smooths_plot=True, context_plot=context_plot, fillers=fillers
    )
    save_agg_symbols_df(smooths_p_vals, smooths_p_vals_path)
    draw_on_smooths_img(smooth_img_path, smooths_p_vals)


def add_p_vals_symbols_context_reread_plot(smooth_img_path, smooths_p_vals_path):
    # Read the CSV file
    smooths_p_vals = pd.read_csv(smooths_p_vals_path)
    # Add x,y columns, symbol columns
    logger.info("Smooths Plot | ----------------------")
    smooths_p_vals = get_smooths_p_vals_symbols(smooths_p_vals, smooths_p_vals_path)
    smooths_p_vals = smooths_p_vals[
        (smooths_p_vals["reread_condition"] == 1)
        & (smooths_p_vals["context_type"] == "regime_context")
    ].reset_index()
    smooths_p_vals = add_x_y_vals(smooths_p_vals, context_reread_plot=True)
    save_agg_symbols_df(smooths_p_vals, smooths_p_vals_path)
    draw_on_smooths_img(smooth_img_path, smooths_p_vals, context_reread_plot=True)
    return smooths_p_vals


def create_grid(images, names, result_path, title=None, cols=None, rows=None):
    num_images = len(images)

    # Determine the grid size (rows and columns)
    if not cols:
        cols = min(math.ceil(math.sqrt(num_images)), 4)
    if not rows:
        rows = math.ceil(num_images / cols)

    # Get the size of each individual image
    width, height = images[0][0].size

    # Add the name above the image
    font_path = (
        "DejaVuSans-Bold.ttf"  # Replace with the path to your downloaded font file
    )
    font_size = 130
    font = ImageFont.truetype(font_path, font_size)

    # Calculate the additional height needed for text
    text_height = 200  # Height of a capital letter

    # Calculate the additional height needed for the title
    title_height = 300 if title else 0

    # Create a new image with the appropriate size
    grid_image = Image.new(
        "RGB",
        (cols * width, rows * (height + text_height) + title_height),
        (255, 255, 255),
    )

    # Create a draw object
    draw = ImageDraw.Draw(grid_image)

    # Add the title above all images
    if title:
        title_x = (
            cols * width - len(title) * font_size
        ) // 2  # Center the title horizontally
        title_y = 0
        draw.text((title_x, title_y), title, font=font, fill="black")

    # Paste each image into the grid and add text
    for i, (img, name) in enumerate(zip(images, names)):
        x = (i % cols) * width
        y = (i // cols) * (height + text_height) + title_height

        # Add the name above the image
        if len(max(names, key=len)) < 20:
            text_x = x + 0.8 * (
                (width - font_size) // 2
            )  # Center the text horizontally
        else:
            text_x = x + 0.3 * ((width - font_size) // 2)
        text_y = y
        draw.text((text_x, text_y), name, font=font, fill="black")

        # Paste the image below the text
        img_y = y + text_height
        grid_image.paste(img[0], (x, img_y))
    grid_image.save(result_path)


def get_image_dir_by_analysis_type(analysis_type, results_dir, et_data_name):
    if analysis_type == "Basic_Analysis":
        return f"{results_dir}/results/{et_data_name}"
    elif analysis_type == "Critical_Span":
        return f"{results_dir}/results_critical_span/{et_data_name}"
    elif analysis_type == "Consecutive_Repeated_reading":
        return f"{results_dir}/results_reread_split/{et_data_name}"
    elif analysis_type == "Different_Surprisal_Estimates":
        return f"{results_dir}/results_different_surp_estimates/{et_data_name}"
    else:
        raise ValueError(f"got unexpected analysis_type: {analysis_type}")


def get_files(image_dir, model_name):
    all_files = os.listdir(image_dir)

    if model_name != "all_models":
        linear_comp_files = [
            f
            for f in all_files
            if (f.endswith("_linear_comp_df.pdf") and (model_name in f))
        ]
        surp_link_files = [
            f for f in all_files if (f.endswith("_surp_link.pdf") and (model_name in f))
        ]
        dll_p_vals_files = [
            f
            for f in all_files
            if (f.endswith("_dll_p_vals_df.csv") and (model_name in f))
        ]
        smooths_p_vals_files = [
            f
            for f in all_files
            if (f.endswith("_smooths_p_vals.csv") and (model_name in f))
        ]
    else:
        linear_comp_files = [
            f for f in all_files if (f.endswith("_linear_comp_df.pdf"))
        ]
        surp_link_files = [f for f in all_files if (f.endswith("_surp_link.pdf"))]
        dll_p_vals_files = [f for f in all_files if (f.endswith("_dll_p_vals_df.csv"))]
        smooths_p_vals_files = [
            f for f in all_files if (f.endswith("_smooths_p_vals.csv"))
        ]

    return linear_comp_files, surp_link_files, dll_p_vals_files, smooths_p_vals_files


def run_p_vals(model_name, analysis_type, results_dir, et_data_name, only_grid):
    image_dir = get_image_dir_by_analysis_type(analysis_type, results_dir, et_data_name)

    # Filter files that end with '_linear_comp_df.pdf' and '_p_vals_df.csv'
    linear_comp_files, surp_link_files, dll_p_vals_files, smooths_p_vals_files = (
        get_files(image_dir, model_name)
    )

    # Extract the prefix from the file names
    def get_prefix(file_name, suffix):
        return file_name[: -len(suffix)]

    # Create a dictionary to map prefixes to file paths
    linear_comp_dict = {
        get_prefix(f, "_linear_comp_df.pdf"): os.path.join(image_dir, f)
        for f in linear_comp_files
    }
    surp_link_dict = {
        get_prefix(f, "_surp_link.pdf"): os.path.join(image_dir, f)
        for f in surp_link_files
    }
    dll_p_vals_dict = {
        get_prefix(f, "_dll_p_vals_df.csv"): os.path.join(image_dir, f)
        for f in dll_p_vals_files
    }
    smooths_p_vals_dict = {
        get_prefix(f, "_smooths_p_vals.csv"): os.path.join(image_dir, f)
        for f in smooths_p_vals_files
    }

    # Find common prefixes and process the files
    for prefix in linear_comp_dict:
        if prefix in dll_p_vals_dict:
            logger.info(f"Add p vals | {prefix}")
            dll_img_path = linear_comp_dict[prefix]
            dll_p_vals_path = dll_p_vals_dict[prefix]
            prefix2 = prefix.replace("_useCV=TRUE", "_additive=TRUE")
            smooth_img_path = surp_link_dict[prefix2]
            smooths_p_vals_path = smooths_p_vals_dict[prefix2]
            if not only_grid:
                add_p_vals_symbols(
                    dll_img_path, dll_p_vals_path, smooth_img_path, smooths_p_vals_path
                )

    return dll_p_vals_files


def get_grid_title_by_analysis_type(analysis_type, model_name_title):
    if analysis_type == "Basic_Analysis":
        return f"{model_name_title}"
    elif analysis_type == "Critical_Span":
        return f"{model_name_title} | Information Seeking"
    elif analysis_type == "Consecutive_Repeated_reading":
        return f"{model_name_title} | Repeated Reading"
    elif analysis_type == "Different_Surprisal_Estimates":
        return "Different_Surprisal_Estimates"
    else:
        raise ValueError(f"got unexpected analysis_type: {analysis_type}")


def get_files_for_grid(with_p_dir, model_name):
    all_files = os.listdir(with_p_dir)

    if model_name != "all_models":
        linear_comp_files = [
            f
            for f in all_files
            if (f.endswith("_linear_comp_df.pdf") and (model_name in f))
        ]
        surp_link_files = [
            f for f in all_files if (f.endswith("_surp_link.pdf") and (model_name in f))
        ]
    else:
        linear_comp_files = [
            f for f in all_files if (f.endswith("_linear_comp_df.pdf"))
        ]
        surp_link_files = [f for f in all_files if (f.endswith("_surp_link.pdf"))]
    return linear_comp_files, surp_link_files


def find_files_by_pattern(
    linear_comp_dict, dll_p_vals_dict, surp_link_dict, title, model_name
):
    surp_link_images = []
    dll_images = []
    names = []
    for prefix in linear_comp_dict:
        if prefix in dll_p_vals_dict:
            dll_path = linear_comp_dict[prefix]
            prefix2 = prefix.replace("_useCV=TRUE", "_additive=TRUE")
            surp_path = surp_link_dict[prefix2]

            _, file_name = os.path.split(dll_path)
            if (
                title == "Different_Surprisal_Estimates"
            ):  # Extract model names from the file names
                model_pattern = re.compile(r"FirstPassGD - (.+?)-Surprisal")
            elif title == "GPT2 small | Different Contexts":
                model_pattern = re.compile(
                    r"Context-(.+?)_re=FALSE"
                )  # Extract context type
            elif model_name == "gpt2":
                model_pattern = re.compile(
                    r"(.+?) - gpt2-Surprisal"
                )  # Extract eye metric
            elif model_name == "EleutherAI-pythia-70m":
                model_pattern = re.compile(
                    r"(.+?) - EleutherAI-pythia-70m-Surprisal"
                )  # Extract eye metric
            elif model_name == "EleutherAI-pythia-160m":
                model_pattern = re.compile(
                    r"(.+?) - EleutherAI-pythia-160m-Surprisal"
                )  # Extract eye metric
            else:
                logger.info("Grid Plot| missing re model_pattern")

            match = model_pattern.search(file_name)
            if match:
                dll_images.append(convert_from_path(dll_path))
                surp_link_images.append(convert_from_path(surp_path))
                names.append(match.group(1))

    logger.info(f"Grid| names: {names}")
    logger.info(
        f"Grid| n images: {len(surp_link_images)}, {len(dll_images)}, {len(names)}"
    )
    return names, surp_link_images, dll_images


def _add_ppl_to_names(names):
    # replace names
    names_df = pd.read_csv(PPL_PATH)

    def get_new_name(name):
        return names_df.loc[names_df["model_id"] == name, "name_with_ppl"].item()

    names = [get_new_name(n) for n in names]
    logger.info(f"Grid| ppl names: {names}")
    return names


def _different_estimates_filter(names, surp_link_images, dll_images):
    logger.info(f"before filter: {len(names)}")
    filtered_surp_link_images = [
        img for name, img in zip(names, surp_link_images) if name != "c"
    ]
    filtered_dll_images = [img for name, img in zip(names, dll_images) if name != "c"]
    logger.info(f"after filter: {len(names)}")
    return names, filtered_surp_link_images, filtered_dll_images


def run_different_estimates_grid(
    names, surp_link_images, dll_images, title, with_p_dir
):
    names, surp_link_images, dll_images = _different_estimates_filter(
        names, surp_link_images, dll_images
    )
    names = _add_ppl_to_names(names)

    create_grid(
        dll_images,
        names,
        title=title,
        result_path=os.path.join(with_p_dir, f"grid_plot_{title}_dll_part.png"),
        cols=6,
        rows=5,
    )
    create_grid(
        surp_link_images,
        names,
        title=title,
        result_path=os.path.join(with_p_dir, f"grid_plot_{title}_gam_fits.png"),
        cols=6,
        rows=5,
    )

    # names_first_half = names[:len(names)//2]
    # names_second_half = names[len(names)//2:]

    # surp_link_images_first_half = surp_link_images[:len(surp_link_images)//2]
    # surp_link_images_second_half = surp_link_images[len(surp_link_images)//2:]

    # dll_images_first_half = dll_images[:len(dll_images)//2]
    # dll_images_second_half = dll_images[len(dll_images)//2:]

    # create_grid(
    #     dll_images_first_half,
    #     names_first_half,
    #     title=title,
    #     result_path=os.path.join(with_p_dir, f"grid_plot_{title}_dll_part_1.png"),
    #     cols=4
    # )
    # create_grid(
    #     dll_images_second_half,
    #     names_second_half,
    #     title=title,
    #     result_path=os.path.join(with_p_dir, f"grid_plot_{title}_dll_part_2.png"),
    #     cols=4
    # )
    # create_grid(
    #     surp_link_images_first_half,
    #     names_first_half,
    #     title=title,
    #     result_path=os.path.join(with_p_dir, f"grid_plot_{title}_gam_fits_part_1.png"),
    #     cols=4
    # )
    # create_grid(
    #     surp_link_images_second_half,
    #     names_second_half,
    #     title=title,
    #     result_path=os.path.join(with_p_dir, f"grid_plot_{title}_gam_fits_part_2.png"),
    #     cols=4
    # )


def _sort_by_custom_order(title, model_name, names, surp_link_images, dll_images):
    # Zip the lists together and Sort by custom order
    if (
        title == "Different_Surprisal_Estimates"
    ):  # Extract model names from the file names
        custom_order = (
            pd.read_csv(PPL_PATH)
            .sort_values(by=["sentence_level_ppl"])["model_id"]
            .tolist()
        )
    elif title == "GPT2 small | Different Contexts":
        custom_order = [
            "p",
            "p-p",
            "fillerP-p",
            "q-p",
            "fillerQ-p",
            "p-q-p",
            "fillerP-fillerQ-p",
        ]
    elif model_name == "gpt2":
        custom_order = ["FirstPassFF", "FF", "FirstPassGD", "GD", "TF"]
    else:
        custom_order = ["FirstPassFF", "FF", "GD", "TF"]
    logger.info(f"Grid| custom_order: {custom_order}")

    # Sort the and filter the data based on the custom order
    order_dict = {name: index for index, name in enumerate(custom_order)}
    filtered_data = [
        (name, surp_link, dll)
        for name, surp_link, dll in zip(names, surp_link_images, dll_images)
        if name in order_dict
    ]
    if title == "Different_Surprisal_Estimates":
        sorted_data = sorted(filtered_data, key=lambda x: -order_dict[x[0]])
    else:
        sorted_data = sorted(filtered_data, key=lambda x: order_dict[x[0]])

    # Unzip the sorted data
    return zip(*sorted_data)


def run_grid(
    dll_p_vals_files,
    model_name,
    model_name_title,
    analysis_type,
    results_dir,
    et_data_name,
):
    # Define the directory containing the image files
    image_dir = get_image_dir_by_analysis_type(analysis_type, results_dir, et_data_name)
    title = get_grid_title_by_analysis_type(analysis_type, model_name_title)

    # Extract the prefix from the file names
    def get_prefix(file_name, suffix):
        return file_name[: -len(suffix)]

    dll_p_vals_dict = {
        get_prefix(f, "_dll_p_vals_df.csv"): os.path.join(image_dir, f)
        for f in dll_p_vals_files
    }

    # files with p value
    with_p_dir = os.path.join(image_dir, "with_p_vals")

    # Filter files that end with '_linear_comp_df.pdf' and '_p_vals_df.csv'
    linear_comp_files, surp_link_files = get_files_for_grid(with_p_dir, model_name)

    # Create a dictionary to map prefixes to file paths
    linear_comp_dict = {
        get_prefix(f, "_linear_comp_df.pdf"): os.path.join(with_p_dir, f)
        for f in linear_comp_files
    }
    surp_link_dict = {
        get_prefix(f, "_surp_link.pdf"): os.path.join(with_p_dir, f)
        for f in surp_link_files
    }

    # Find common prefixes and process the files
    names, surp_link_images, dll_images = find_files_by_pattern(
        linear_comp_dict, dll_p_vals_dict, surp_link_dict, title, model_name
    )

    # sort by custom order
    names, surp_link_images, dll_images = _sort_by_custom_order(
        title, model_name, names, surp_link_images, dll_images
    )
    logger.info(f"Grid| sorted names: {names}")
    logger.info(
        f"Grid| n images: {len(surp_link_images)}, {len(dll_images)}, {len(names)}"
    )

    if title == "Different_Surprisal_Estimates":
        run_different_estimates_grid(
            names, surp_link_images, dll_images, title, with_p_dir
        )
    else:
        create_grid(
            dll_images,
            names,
            title=title,
            result_path=os.path.join(with_p_dir, f"grid_plot_{title}_dll.png"),
        )
        create_grid(
            surp_link_images,
            names,
            title=title,
            result_path=os.path.join(with_p_dir, f"grid_plot_{title}_gam_fits.png"),
        )


def _get_prefix_by_article_level(article_level):
    if article_level == "paragraph":
        return "Context"
    elif article_level == "article":
        return "Article-Context"
    else:
        raise ValueError(f"got unexpected article_level: {article_level}")


def add_anotations_context_plots(
    model_name, results_dir, et_data_name, article_level="paragraph"
):
    # context + zoom in only on rr

    article_prefix = _get_prefix_by_article_level(article_level)

    for fillers in ["TRUE", "FALSE"]:
        logger.info(f"Context Plot| fillers: {fillers}")
        context_dll_plot_path = f"{results_dir}/results_context/{et_data_name}/{article_prefix}-FirstPassGD-{model_name}-zoom_in_only_for_rr-fillers={fillers}-dll.pdf"
        context_GAM_plot_path = f"{results_dir}/results_context/{et_data_name}/{article_prefix}-FirstPassGD-{model_name}-zoom_in_only_for_rr-fillers={fillers}-surp_link.pdf"
        context_dll_p_vals_path = f"{results_dir}/results_context/{et_data_name}/{article_prefix}-FirstPassGD-{model_name}-fillers={fillers}-dll_p_vals_df.csv"
        context_smooths_p_vals_path = f"{results_dir}/results_context/{et_data_name}/{article_prefix}-FirstPassGD-{model_name}-fillers={fillers}-smooths_p_vals.csv"

        add_p_vals_symbols(
            context_dll_plot_path,
            context_dll_p_vals_path,
            context_GAM_plot_path,
            context_smooths_p_vals_path,
            context_plot=True,
            fillers=fillers,
        )


def run_p_vals_and_grid(
    model_name,
    model_name_title,
    analysis_type,
    results_dir,
    et_data_name,
    only_grid=False,
):
    dll_p_vals_files = run_p_vals(
        model_name, analysis_type, results_dir, et_data_name, only_grid
    )
    run_grid(
        dll_p_vals_files,
        model_name,
        model_name_title,
        analysis_type,
        results_dir,
        et_data_name,
    )


def get_large_models_names(lm_path):
    lm_df = pd.read_csv(lm_path)

    def extract_model(name):
        model_pattern = re.compile(r"(.+?)-Surprisal")
        return model_pattern.search(name).group(1)

    lm_df["model"] = lm_df["surp_col"].apply(lambda x: extract_model(x))
    return lm_df["model"].unique()


def test_add_ppl_to_names():
    names = [
        "EleutherAI-pythia-1.4b",
        "mistralai-Mistral-7B-v0.3",
        "EleutherAI-pythia-1b",
        "meta-llama-Llama-2-7b-chat-hf",
        "EleutherAI-gpt-neo-2.7B",
        "EleutherAI-pythia-6.9b",
        "google-gemma-7b",
        "EleutherAI-pythia-70m",
        "meta-llama-Llama-2-13b-chat-hf",
        "EleutherAI-gpt-neo-125M",
        "EleutherAI-pythia-410m",
        "facebook-opt-2.7b",
        "gpt2-large",
        "mistralai-Mistral-7B-Instruct-v0.3",
        "facebook-opt-1.3b",
        "meta-llama-Llama-2-13b-hf",
        "mistralai-Mistral-7B-v0.1",
        "facebook-opt-350m",
        "EleutherAI-gpt-j-6B",
        "EleutherAI-gpt-neo-1.3B",
        "meta-llama-Llama-2-70b-hf",
        "meta-llama-Llama-2-7b-hf",
        "gpt2-xl",
        "EleutherAI-pythia-2.8b",
        "EleutherAI-pythia-160m",
        "gpt2-medium",
        "gpt2",
    ]

    surp_link_images = range(1, len(names))
    dll_images = range(1, len(names))
    names, surp_link_images, dll_images = _add_ppl_to_names(
        names, surp_link_images, dll_images
    )
