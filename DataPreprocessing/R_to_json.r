library(jsonlite)
json_data <- jsonlite::toJSON(EIC_list_14)
write(json_data, "EIC14_JSON.json")