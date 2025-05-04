# Carregar ggplot2 se ainda não estiver carregado
# install.packages("ggplot2")
library(ggplot2)
# Opcional para rótulos de texto melhores: install.packages("ggrepel")
# library(ggrepel)

#' Plot Circumplex Model Scores
#'
#' Creates a 2D plot visualizing Valence and Arousal scores from the
#' output of `song_sent_scores`.
#'
#' @param results_sent A list object returned by the `song_sent_scores` function,
#'   containing `audio_scores` and optionally `text_scores`.
#' @param title Optional: A title for the plot. Default is "Circumplex Model Scores".
#' @param point_size Size of the points on the plot. Default is 4.
#' @param label_points Logical. If TRUE, adds text labels ("Audio", "Text") near the points.
#'   Requires the 'ggrepel' package for better label placement if TRUE. Default is TRUE.
#'
#' @return A ggplot object, or NULL if no valid scores are found.
#' @export
#'
#' @examples
#' \dontrun{
#' # --- Assuming results_sent is the output from song_sent_scores ---
#'
#' # Exemplo 1: Plotar com labels
#' p1 <- plot_circumplex_scores(results_sent)
#' if (!is.null(p1)) print(p1)
#'
#' # Exemplo 2: Plotar sem labels e com título customizado
#' p2 <- plot_circumplex_scores(results_sent,
#'                              title = "Valence/Arousal - 'Gostava Tanto de Você'",
#'                              label_points = FALSE)
#' if (!is.null(p2)) print(p2)
#'
#' # Exemplo 3: Plotar se só tiver audio_scores
#' results_audio_only <- list(
#'    audio_scores = c(neg_valence=0.7, pos_valence=0.3, low_arousal=0.8, high_arousal=0.2),
#'    text_scores = NULL # Simular ausência de scores de texto
#' )
#' p3 <- plot_circumplex_scores(results_audio_only)
#' if (!is.null(p3)) print(p3)
#' }
plot_circumplex_scores <- function(results_sent,
                                   title = "Circumplex Model Scores",
                                   point_size = 4,
                                   label_points = TRUE) {
  
  # --- 1. Validar Input e Extrair Dados ---
  if (!is.list(results_sent) ||
      (is.null(results_sent$audio_scores) && is.null(results_sent$text_scores))) {
    warning("Input 'results_sent' is not a valid list or contains no scores.", call. = FALSE)
    return(NULL)
  }
  
  plot_data_list <- list()
  
  # Processar Audio Scores
  if (!is.null(results_sent$audio_scores) && !all(is.na(results_sent$audio_scores)) && length(results_sent$audio_scores) == 4) {
    scores_audio <- results_sent$audio_scores
    valence_audio <- scores_audio["pos_valence"] - scores_audio["neg_valence"]
    arousal_audio <- scores_audio["high_arousal"] - scores_audio["low_arousal"]
    plot_data_list$audio <- data.frame(
      modality = "Audio",
      valence = as.numeric(valence_audio), # Garantir numérico
      arousal = as.numeric(arousal_audio)
    )
  } else {
    warning("Audio scores missing, invalid, or not of length 4 in results_sent.", call. = FALSE)
  }
  
  # Processar Text Scores
  if (!is.null(results_sent$text_scores) && !all(is.na(results_sent$text_scores)) && length(results_sent$text_scores) == 4) {
    scores_text <- results_sent$text_scores
    valence_text <- scores_text["pos_valence"] - scores_text["neg_valence"]
    arousal_text <- scores_text["high_arousal"] - scores_text["low_arousal"]
    plot_data_list$text <- data.frame(
      modality = "Text",
      valence = as.numeric(valence_text),
      arousal = as.numeric(arousal_text)
    )
  } # Não emitir warning se text_scores for NULL, pois é opcional
  
  # Combinar dados
  if (length(plot_data_list) == 0) {
    warning("No valid audio or text scores found to plot.", call. = FALSE)
    return(NULL)
  }
  plot_data <- do.call(rbind, plot_data_list)
  
  # Remover linhas com NA em valence ou arousal (caso um cálculo falhe)
  plot_data <- na.omit(plot_data)
  if(nrow(plot_data) == 0) {
    warning("No valid (non-NA) coordinates to plot after calculation.", call. = FALSE)
    return(NULL)
  }
  
  
  # --- 2. Criar o Gráfico ggplot ---
  p <- ggplot(plot_data, aes(x = valence, y = arousal)) +
    # Linhas de eixo central
    geom_hline(yintercept = 0, linetype = "dashed", color = "grey60", linewidth = 0.5) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "grey60", linewidth = 0.5) +
    
    # Pontos para cada modalidade
    geom_point(aes(color = modality, shape = modality), size = point_size, alpha = 0.8) +
    
    # Escalas e Limites
    scale_x_continuous(limits = c(-1.1, 1.1), breaks = seq(-1, 1, 0.5), name = "Valence") +
    scale_y_continuous(limits = c(-1.1, 1.1), breaks = seq(-1, 1, 0.5), name = "Arousal") +
    scale_color_manual(values = c("Audio" = "#4c9a9e", "Text" = "#b3bef2"), name = "Modality") +
    scale_shape_manual(values = c("Audio" = 16, "Text" = 17), name = "Modality") + # Ponto sólido e triângulo
    
    # Rótulos e Título
    labs(
      title = title,
      caption = "Valence: -1 (Negative) to +1 (Positive)\nArousal: -1 (Low/Mild) to +1 (High/Intense)"
    ) +
    
    # Tema e Proporção
    theme_classic(base_size = 11) +
    theme(
      aspect.ratio = 1, # Garante gráfico quadrado
      panel.grid.major = element_line(color = "grey90", linewidth = 0.2),
      panel.grid.minor = element_blank(),
      plot.title = element_text(hjust = 0.5, face = "bold"),
      plot.caption = element_text(hjust = 0, size = 8, color = "grey40")
    )
  
  # Adicionar rótulos de texto se solicitado
  if (label_points) {
    # Tentar usar ggrepel se disponível, senão usar geom_text simples
    if (requireNamespace("ggrepel", quietly = TRUE)) {
      p <- p + ggrepel::geom_text_repel(aes(label = modality), size = 3.5, point.padding = 0.5, box.padding = 0.5)
    } else {
      warning("Pacote 'ggrepel' não instalado. Usando geom_text para labels (podem sobrepor).", call. = FALSE, immediate. = TRUE)
      p <- p + geom_text(aes(label = modality), size = 3.5, vjust = -1.2) # Ajuste vertical simples
    }
  }
  
  return(p)
}
