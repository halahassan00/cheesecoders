import { useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card"
import { Button } from "./ui/button"
import { Badge } from "./ui/badge"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "./ui/table"
import { Input } from "./ui/input"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./ui/select"
import { BookmarkCheck, Download, Search, Filter, Trash2, Eye } from "lucide-react"

// Mock saved results data
const savedResults = [
  {
    id: "SR-001",
    name: "Kepler-442b Analysis",
    type: "Classification",
    date: "2023-10-15",
    classification: "Super Earth",
    confidence: 87.3,
    habitability: 0.72,
    status: "Confirmed"
  },
  {
    id: "SR-002", 
    name: "K2-18b Transit Study",
    type: "Light Curve",
    date: "2023-10-14",
    classification: "Sub-Neptune",
    confidence: 92.1,
    habitability: 0.58,
    status: "Under Review"
  },
  {
    id: "SR-003",
    name: "TOI-715b Detection",
    type: "Classification",
    date: "2023-10-13",
    classification: "Rocky Planet",
    confidence: 78.9,
    habitability: 0.84,
    status: "Confirmed"
  },
  {
    id: "SR-004",
    name: "TRAPPIST-1e Analysis",
    type: "System Study",
    date: "2023-10-12",
    classification: "Rocky Planet",
    confidence: 95.7,
    habitability: 0.91,
    status: "Published"
  },
  {
    id: "SR-005",
    name: "Proxima Centauri b",
    type: "Classification",
    date: "2023-10-11",
    classification: "Rocky Planet",
    confidence: 89.4,
    habitability: 0.67,
    status: "Confirmed"
  },
  {
    id: "SR-006",
    name: "WASP-121b Study",
    type: "Atmospheric",
    date: "2023-10-10",
    classification: "Hot Jupiter",
    confidence: 99.2,
    habitability: 0.01,
    status: "Published"
  }
]

export function SavedResults() {
  const [searchTerm, setSearchTerm] = useState("")
  const [filterType, setFilterType] = useState("all")
  const [filterStatus, setFilterStatus] = useState("all")

  const filteredResults = savedResults.filter(result => {
    const matchesSearch = result.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         result.classification.toLowerCase().includes(searchTerm.toLowerCase())
    const matchesType = filterType === "all" || result.type === filterType
    const matchesStatus = filterStatus === "all" || result.status === filterStatus
    
    return matchesSearch && matchesType && matchesStatus
  })

  const getStatusBadge = (status: string) => {
    switch (status) {
      case "Confirmed":
        return <Badge className="bg-green-600 text-white">Confirmed</Badge>
      case "Under Review":
        return <Badge variant="secondary">Under Review</Badge>
      case "Published":
        return <Badge className="bg-blue-600 text-white">Published</Badge>
      default:
        return <Badge variant="outline">{status}</Badge>
    }
  }

  const getClassificationBadge = (classification: string) => {
    const colors: { [key: string]: string } = {
      "Super Earth": "bg-green-600",
      "Rocky Planet": "bg-orange-600", 
      "Sub-Neptune": "bg-blue-600",
      "Hot Jupiter": "bg-red-600"
    }
    
    return (
      <Badge className={`${colors[classification] || "bg-gray-600"} text-white`}>
        {classification}
      </Badge>
    )
  }

  const getHabitabilityColor = (score: number) => {
    if (score >= 0.7) return "text-green-400"
    if (score >= 0.4) return "text-yellow-400"
    return "text-red-400"
  }

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <Card className="bg-slate-800/50 border-slate-700">
        <CardHeader>
          <CardTitle className="text-white flex items-center gap-3">
            <BookmarkCheck className="h-6 w-6 text-blue-400" />
            Saved Analysis Results
            <Badge variant="secondary">{filteredResults.length} results</Badge>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-slate-300">
            Access and manage your saved exoplanet analysis results, classifications, and research findings.
          </p>
        </CardContent>
      </Card>

      {/* Filters and Search */}
      <Card className="bg-slate-800/50 border-slate-700">
        <CardContent className="pt-6">
          <div className="flex flex-wrap gap-4 items-end">
            <div className="flex-1 min-w-64">
              <label className="block text-sm text-slate-300 mb-2">Search Results</label>
              <div className="relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-slate-400 h-4 w-4" />
                <Input
                  placeholder="Search by name or classification..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="bg-slate-700 border-slate-600 text-white pl-10"
                />
              </div>
            </div>
            
            <div className="w-48">
              <label className="block text-sm text-slate-300 mb-2">Analysis Type</label>
              <Select value={filterType} onValueChange={setFilterType}>
                <SelectTrigger className="bg-slate-700 border-slate-600 text-white">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent className="bg-slate-700 border-slate-600">
                  <SelectItem value="all">All Types</SelectItem>
                  <SelectItem value="Classification">Classification</SelectItem>
                  <SelectItem value="Light Curve">Light Curve</SelectItem>
                  <SelectItem value="System Study">System Study</SelectItem>
                  <SelectItem value="Atmospheric">Atmospheric</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="w-48">
              <label className="block text-sm text-slate-300 mb-2">Status</label>
              <Select value={filterStatus} onValueChange={setFilterStatus}>
                <SelectTrigger className="bg-slate-700 border-slate-600 text-white">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent className="bg-slate-700 border-slate-600">
                  <SelectItem value="all">All Status</SelectItem>
                  <SelectItem value="Confirmed">Confirmed</SelectItem>
                  <SelectItem value="Under Review">Under Review</SelectItem>
                  <SelectItem value="Published">Published</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <Button variant="outline" className="border-slate-600 text-slate-300 hover:bg-slate-700">
              <Filter className="mr-2 h-4 w-4" />
              Advanced
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Results Table */}
      <Card className="bg-slate-800/50 border-slate-700">
        <CardContent className="pt-6">
          <Table>
            <TableHeader>
              <TableRow className="border-slate-700">
                <TableHead className="text-slate-300">ID</TableHead>
                <TableHead className="text-slate-300">Name</TableHead>
                <TableHead className="text-slate-300">Type</TableHead>
                <TableHead className="text-slate-300">Date</TableHead>
                <TableHead className="text-slate-300">Classification</TableHead>
                <TableHead className="text-slate-300">Confidence</TableHead>
                <TableHead className="text-slate-300">Habitability</TableHead>
                <TableHead className="text-slate-300">Status</TableHead>
                <TableHead className="text-slate-300">Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {filteredResults.map((result) => (
                <TableRow key={result.id} className="border-slate-700 hover:bg-slate-700/50">
                  <TableCell className="text-slate-300">{result.id}</TableCell>
                  <TableCell className="text-white">{result.name}</TableCell>
                  <TableCell className="text-slate-300">{result.type}</TableCell>
                  <TableCell className="text-slate-300">{result.date}</TableCell>
                  <TableCell>{getClassificationBadge(result.classification)}</TableCell>
                  <TableCell className="text-white">{result.confidence}%</TableCell>
                  <TableCell className={getHabitabilityColor(result.habitability)}>
                    {result.habitability}
                  </TableCell>
                  <TableCell>{getStatusBadge(result.status)}</TableCell>
                  <TableCell>
                    <div className="flex gap-2">
                      <Button size="sm" variant="outline" className="border-slate-600 text-slate-300 hover:bg-slate-700">
                        <Eye className="h-4 w-4" />
                      </Button>
                      <Button size="sm" variant="outline" className="border-slate-600 text-slate-300 hover:bg-slate-700">
                        <Download className="h-4 w-4" />
                      </Button>
                      <Button size="sm" variant="outline" className="border-slate-600 text-red-400 hover:bg-red-900/20">
                        <Trash2 className="h-4 w-4" />
                      </Button>
                    </div>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      {/* Quick Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <Card className="bg-slate-800/50 border-slate-700">
          <CardContent className="pt-6 text-center">
            <div className="text-3xl text-blue-400 mb-2">24</div>
            <div className="text-slate-400">Total Results</div>
          </CardContent>
        </Card>
        
        <Card className="bg-slate-800/50 border-slate-700">
          <CardContent className="pt-6 text-center">
            <div className="text-3xl text-green-400 mb-2">18</div>
            <div className="text-slate-400">Confirmed</div>
          </CardContent>
        </Card>
        
        <Card className="bg-slate-800/50 border-slate-700">
          <CardContent className="pt-6 text-center">
            <div className="text-3xl text-yellow-400 mb-2">4</div>
            <div className="text-slate-400">Under Review</div>
          </CardContent>
        </Card>
        
        <Card className="bg-slate-800/50 border-slate-700">
          <CardContent className="pt-6 text-center">
            <div className="text-3xl text-purple-400 mb-2">7</div>
            <div className="text-slate-400">Published</div>
          </CardContent>
        </Card>
      </div>

      {/* Bulk Actions */}
      <Card className="bg-slate-800/50 border-slate-700">
        <CardContent className="pt-6">
          <div className="flex gap-4">
            <Button className="bg-blue-600 hover:bg-blue-700 text-white">
              <Download className="mr-2 h-4 w-4" />
              Export All Results
            </Button>
            <Button variant="outline" className="border-slate-600 text-slate-300 hover:bg-slate-700">
              Backup to Archive
            </Button>
            <Button variant="outline" className="border-slate-600 text-slate-300 hover:bg-slate-700">
              Generate Report
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
