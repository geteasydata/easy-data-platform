
// Power Query M Code for Data Loading
let
    Source = Csv.Document(File.Contents("data.csv"), [Delimiter=",", Encoding=65001]),
    #"Promoted Headers" = Table.PromoteHeaders(Source, [PromoteAllScalars=true]),
    #"Changed Type" = Table.TransformColumnTypes(#"Promoted Headers", {
        {"Date", type text},
        {"Time", type text},
        {"Booking ID", type text},
        {"Booking Status", type text},
        {"Customer ID", type text},
        {"Vehicle Type", type text},
        {"Pickup Location", type text},
        {"Drop Location", type text},
        {"Avg VTAT", type number},
        {"Avg CTAT", type number},
    })
in
    #"Changed Type"
